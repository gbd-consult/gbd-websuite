"""Interface with GekoS-Bau software."""

import gws
import gws.base.action
import gws.base.db
import gws.base.template
import gws.ext.db.provider.postgres
import gws.ext.helper.alkis
import gws.gis.proj
import gws.gis.shape
import gws.web.error

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

class SourceConfig(t.Config):
    url: t.Url  #: gek-online base url
    params: dict  #: additional parameters for gek-online calls
    instance: str #: instance name for this source


class Config(t.WithTypeAndAccess):
    """GekoS action"""

    crs: t.Crs = ''  #: CRS for gekos data
    db: t.Optional[str]  #: database provider uid
    sources: t.Optional[t.List[SourceConfig]]  #: gek-online instances
    position: t.Optional[PositionConfig]  #: position correction for points
    table: gws.base.db.SqlTableConfig  #: sql table configuration
    templates: t.Optional[t.List[t.ext.template.Config]]  #: feature formatting templates


_DEFAULT_TEMPLATES = [
    t.Config(
        subject='feature.title',
        type='html',
        text='{vollnummer}'
    ),
    t.Config(
        subject='feature.teaser',
        type='html',
        text='Flurstück {vollnummer}',
    )
]


class Object(gws.base.action.Object):
    def configure(self):
        super().configure()

        self.alkis = t.cast(gws.ext.helper.alkis.Object, self.root.find_first('gws.ext.helper.alkis'))
        self.crs: t.Crs = self.var('crs')
        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.base.db.require_provider(self, 'gws.ext.db.provider.postgres'))
        self.templates: t.List[t.ITemplate] = gws.base.template.bundle(self, self.var('templates'), _DEFAULT_TEMPLATES)

    def api_find_fs(self, req: t.IRequest, p: GetFsParams) -> GetFsResponse:
        if not self.alkis:
            raise gws.web.error.NotFound()

        feature = self._find_alkis_feature(p)

        if not feature:
            raise gws.web.error.NotFound()

        project = req.require_project(p.projectUid)
        feature.transform_to(project.map.crs)

        f = feature.apply_templates(self.templates).props
        f.attributes = []

        return GetFsResponse(feature=f)

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
        recs = []

        for source in self.var('sources'):
            options = t.Data(
                crs=self.crs,
                url=source.url,
                params=source.params,
                position=self.var('position')
            )
            gr = request.GekosRequest(options, source.instance, cache_lifetime=0)
            rs = gr.run()
            gws.log.info(f'loaded {len(rs)} records from {source.instance!r}')
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
            }, self.crs)

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
