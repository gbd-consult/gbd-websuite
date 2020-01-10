"""Interface with GekoS-Bau software."""

import gws
import gws.common.db
import gws.common.layer
import gws.common.template
import gws.config
import gws.ext.db.provider.postgres
import gws.gis.feature
import gws.gis.proj
import gws.tools.net
import gws.web

import gws.types as t

import gws.ext.tool.alkis

from . import request

"""

    see https://www.gekos.de/
    
    GekoS settings for gws (Verfahrensadministration/GIS Schnittstelle)
    
    base address:
    
        GIS-URL-Base  = http://my-server

    client-side call, handled in the client:
    
        GIS-URL-ShowXY  = /project/my-project/?x=<x>&y=<y>&z=my-scale-value
        
    client-side call, handled in the client
    
        GIS-URL-ShowFs = /project/my-project/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        
    client-side call, handled in the client:
        
        GIS-URL-GetXYFromMap = /project/my-project?&x=<x>&y=<y>&gekosUrl=<returl>
    
    callback urls, handled here
            
        GIS-URL-GetXYFromFs   = /_/?cmd=gekosHttpGetXy&alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        GIS-URL-GetXYFromGrd  = /_/?cmd=gekosHttpGetXy&alkisAd=<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

    NB: the order of placeholders must match _COMBINED_FS_PARAMS and _COMBINED_AD_PARAMS in ext.tool.alkis

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

        self.db: gws.ext.db.provider.postgres = None
        self.alkis: gws.ext.tool.alkis.Object = None
        self.feature_format: t.IFormat = None
        self.crs = ''

    def configure(self):
        super().configure()
        self.crs = self.var('crs')
        self.db: gws.ext.db.provider.postgres.Object = gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres')

        self.alkis: gws.ext.tool.alkis.Object = self.find_first('gws.ext.tool.alkis')

        self.feature_format = self.add_child('gws.common.format', self.var('featureFormat') or DEFAULT_FORMAT)

    def api_find_fs(self, req: t.IRequest, p: GetFsParams) -> GetFsResponse:
        if not self.alkis:
            raise gws.web.error.NotFound()

        f = self._find_alkis_feature(p)

        if not f:
            raise gws.web.error.NotFound()

        return GetFsResponse(feature=f)

    def http_get_xy(self, req: t.IRequest, p: GetFsParams) -> t.HttpResponse:
        if not self.alkis:
            return _text('error:')

        f = self._find_alkis_feature(p)

        if not f:
            gws.log.warn(f'gekos: not found {req.params!r}')
            return _text('error:')

        return _text('%.3f;%.3f' % (f.attr('x'), f.attr('y')))

    def _find_alkis_feature(self, p: GetFsParams):
        res = None
        if p.alkisFs:
            res = self.alkis.find_flurstueck_combined(p.alkisAd)
        elif p.alkisAd:
            res = self.alkis.find_adresse_combined(p.alkisAd)
        if res and res.features:
            return res.features[0]

    def load_data(self):
        """Load gekos data into a postgres table."""

        # @TODO typing, the db provider here is specifically postgres
        # @TODO the whole WKT business is because we don't support EWKB and cannot use batch_insert of geoms

        features = self._get_data_from_gekos()

        gws.log.info(f'saving...')
        count = self._write_gekos_features(features)
        gws.log.info(f'saved {count} records')

    def _get_data_from_gekos(self):

        options = t.Data({
            'crs': self.crs,
            'url': self.var('url'),
            'params': self.var('params'),
            'position': self.var('position'),
        })

        features = []

        for instance in self.var('instances', default=['none']):
            gr = request.GekosRequest(options, instance, cache_lifetime=0)
            fs = gr.run()
            gws.log.info(f'loaded {len(fs)} records from {instance!r}')
            features.extend(fs)

        return features

    def _write_gekos_features(self, features):

        uids = set()
        recs = []

        for f in features:
            if f.uid in uids:
                gws.log.warn(f'non-unique uid={f.uid!r} ignored')
                continue
            uids.add(f.uid)
            rec = f.attributes
            rec['uid'] = f.uid
            rec['wkt_geometry'] = f.shape.wkt
            recs.append(rec)

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
                        "wkt_geometry" TEXT,
                        "wkb_geometry" geometry(POINT, {srid})
                    )
                ''',
                f'''CREATE INDEX {name}_antragsart ON {schema}.{name} USING btree("AntragsartID")''',
                f'''CREATE INDEX {name}_geom ON {schema}.{name} USING GIST(wkb_geometry)'''
            ]

            for s in sql:
                conn.exec(s)

            conn.batch_insert(table_name, recs, page_size=2000)
            conn.exec(f'''
                UPDATE {table_name}
                SET wkb_geometry=ST_GeomFromText(wkt_geometry, {srid})
            ''')

            return conn.count(table_name)


def _text(s):
    return t.HttpResponse({
        'mime': 'text/plain',
        'content': s
    })
