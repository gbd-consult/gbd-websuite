"""Interface with GekoS-Bau software."""

import gws
import gws.common.db
import gws.common.template
import gws.ext.db.provider.postgres
import gws.ext.helper.alkis
import gws.gis.proj
import gws.gis.shape
import gws.web

import gws.types as t


from . import request

"""

    see https://www.gekos.de/
    
    GekoS settings for gws (Verfahrensadministration/GIS Schnittstelle)
    
    base address:
    
        GIS-URL-Base  = http://my-server

    client-side call, handled in the client:
    
        GIS-URL-ShowXY  = /project/my-project/?x=<x>&y=<y>&z=my-scale-value
        
    client-side call, handled in the client:
        
        GIS-URL-GetXYFromMap = /project/my-project?&x=<x>&y=<y>&gekosUrl=<returl>
    
    client-side call, redirects here to api_find_fs
    
        GIS-URL-ShowFs = /project/my-project/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        
    callback urls, handled here
            
        GIS-URL-GetXYFromFs   = /_/?cmd=gekosHttpGetXy&alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        GIS-URL-GetXYFromGrd  = /_/?cmd=gekosHttpGetXy&alkisAd=<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

    NB: the order of placeholders must match _COMBINED_FS_PARAMS and _COMBINED_AD_PARAMS in ext.helper.alkis

"""


class GetFsParams(t.Params):
    alkisFs: t.Optional[str]
    alkisAd: t.Optional[str]


class GetFsResponse(t.Response):
    feature: t.FeatureProps


class PositionConfig(t.Config):
    offsetX: int  #: x-offset for points
    offsetY: int  #: y-offset for points
    distance: int = 0  #: radius for points repelling
    angle: int = 0  #: angle for points repelling


class Config(t.WithTypeAndAccess):
    """GekoS action"""

    crs: t.Crs = ''  #: CRS for gekos data
    db: t.Optional[str]  #: database provider uid
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: template for the Flurstueck details popup
    instances: t.Optional[t.List[str]]  #: gek-online instances
    params: dict  #: additional parameters for gek-online calls
    position: t.Optional[PositionConfig]  #: position correction for points
    table: gws.common.db.SqlTableConfig  #: sql table configuration
    url: t.Url  #: gek-online base url


DEFAULT_FORMAT = gws.common.template.FeatureFormatConfig(
    title=gws.common.template.Config(type='html', text='{vollnummer}'),
    teaser=gws.common.template.Config(type='html', text='Flurstück {vollnummer}'),
)


class Object(gws.ActionObject):
    def __init__(self):
        super().__init__()

        self.db: gws.ext.db.provider.postgres.Object = None
        self.alkis: gws.ext.helper.alkis.Object = None
        self.feature_format: t.IFormat = None
        self.crs = ''

    def configure(self):
        super().configure()

        self.crs = self.var('crs')
        self.db: gws.ext.db.provider.postgres.Object = gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres')
        self.alkis: gws.ext.helper.alkis.Object = self.find_first('gws.ext.helper.alkis')
        self.feature_format = self.add_child('gws.common.format', self.var('featureFormat') or DEFAULT_FORMAT)

    def api_find_fs(self, req: t.IRequest, p: GetFsParams) -> GetFsResponse:
        if not self.alkis:
            raise gws.web.error.NotFound()

        f = self._find_alkis_feature(p)

        if not f:
            raise gws.web.error.NotFound()

        return GetFsResponse(feature=f.apply_format(self.feature_format).props)

    def http_get_xy(self, req: t.IRequest, p: GetFsParams) -> t.HttpResponse:
        if not self.alkis:
            return t.HttpResponse(mime='text/plain', content='error:')

        f = self._find_alkis_feature(p)

        if not f:
            gws.log.warn(f'gekos: not found {req.params!r}')
            return t.HttpResponse(mime='text/plain', content='error:')

        return t.HttpResponse(
            mime='text/plain',
            content='%.3f;%.3f' % (f.attr('x'), f.attr('y')))

    def _find_alkis_feature(self, p: GetFsParams):
        res = None
        if p.alkisFs:
            res = self.alkis.find_flurstueck_combined(p.alkisFs)
        elif p.alkisAd:
            res = self.alkis.find_adresse_combined(p.alkisAd)
        if res and res.features:
            return res.features[0]

    def load_data(self):
        """Load gekos data into a postgres table."""

        gws.log.info(f'loading...')
        recs = self._load_gekos_data()

        gws.log.info(f'saving...')
        count = self._write_gekos_data(recs)

        gws.log.info(f'saved {count} records')

    def _load_gekos_data(self):

        options = t.Data(
            crs=self.crs,
            url=self.var('url'),
            params=self.var('params'),
            position=self.var('position'))

        recs = []

        for instance in self.var('instances', default=['none']):
            gr = request.GekosRequest(options, instance, cache_lifetime=0)
            rs = gr.run()
            gws.log.info(f'loaded {len(rs)} records from {instance!r}')
            recs.extend(rs)

        return recs

    def _write_gekos_data(self, recs):

        uids = set()
        buf = []

        for rec in recs:
            if rec['uid'] in uids:
                gws.log.warn(f"non-unique uid={rec['uid']!r} ignored")
                continue
            uids.add(rec['uid'])

            shape = gws.gis.shape.from_geometry({
                'type': 'Point',
                'coordinates': rec.pop('xy')
            }, self.var('crs'))

            rec['wkb_geometry'] = shape.ewkb_hex
            buf.append(rec)

        table_name = self.var('table').name
        srid = gws.gis.proj.as_srid(self.crs)

        with self.db.connect() as conn:

            schema, name = conn.schema_and_table(table_name)

            sql = [
                f'''DROP TABLE IF EXISTS {schema}.{name} CASCADE''',
                f'''
                    CREATE TABLE {schema}.{name} (
                        "uid" CHARACTER VARYING PRIMARY KEY,
                        "ObjectID" CHARACTER VARYING,
                        "AntragsartBez" CHARACTER VARYING,
                        "AntragsartID" INT,
                        "Darstellung" CHARACTER VARYING,
                        "Massnahme" CHARACTER VARYING,
                        "SystemNr" CHARACTER VARYING,
                        "status" CHARACTER VARYING,
                        "Tooltip" CHARACTER VARYING,
                        "UrlFV" CHARACTER VARYING,
                        "UrlOL" CHARACTER VARYING,
                        "Verfahren" CHARACTER VARYING,
                        "instance" CHARACTER VARYING,
                        "wkb_geometry" geometry(POINT, {srid})
                    )
                ''',
                f'''CREATE INDEX {name}_antragsart ON {schema}.{name} USING btree("AntragsartID")''',
                f'''CREATE INDEX {name}_geom ON {schema}.{name} USING GIST(wkb_geometry)'''
            ]

            for s in sql:
                conn.execute(s)

            conn.insert_many(table_name, buf, page_size=2000)

            return conn.count(table_name)
