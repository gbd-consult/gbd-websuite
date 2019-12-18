"""Interface with GekoS-Bau software."""

import gws
import gws.auth.api
import gws.web
import gws.config
import gws.tools.net
import gws.gis.feature
import gws.common.layer
import gws.gis.proj

import gws.types as t

import gws.ext.action.alkis

from . import request

"""

    see https://www.gekos.de/
    
    GekoS settings for gws (Verfahrensadministration/GIS Schnittstelle)
    
    base address:
    
        GIS-URL-Base  = http://my-server

    client-side call, handled in the client:
    
        GIS-URL-ShowXY  = /project/my-project/?x=<x>&y=<y>&z=my-scale-value
        
    client-side call, handled in the client and by the alkis ext:
    
        GIS-URL-ShowFs = /project/my-project/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        
    client-side call, handled in the client:
        
        GIS-URL-GetXYFromMap = /project/my-project?&x=<x>&y=<y>&gekosUrl=<returl>
    
    callback urls, handled here
            
        GIS-URL-GetXYFromFs   = /_/?cmd=gekosHttpGetXy&alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        GIS-URL-GetXYFromGrd  = /_/?cmd=gekosHttpGetXy&alkisAd=<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

    NB: the order of placeholders must match _COMBINED_FS_PARAMS and _COMBINED_AD_PARAMS in ext.action.alkis

"""


class GetXYParams(t.Params):
    alkisFs: str = ''
    alkisAd: str = ''


class PositionConfig(t.Config):
    offsetX: int  #: x-offset for points
    offsetY: int  #: y-offset for points
    distance: int = 0  #: radius for points repelling
    angle: int = 0  #: angle for points repelling


class Config(t.WithTypeAndAccess):
    """GekoS action"""

    crs: t.crsref = ''  #: CRS for gekos data
    db: t.Optional[str]  #: database provider uid
    instances: t.Optional[t.List[str]]  #: gek-online instances
    params: dict  #: additional parameters for gek-online calls
    position: t.Optional[PositionConfig]  #: position correction for points
    table: t.SqlTableConfig  #: sql table configuration
    url: t.url  #: gek-online base url


class Object(gws.ActionObject):
    def configure(self):
        super().configure()
        self.crs = self.var('crs')

        self.db: t.DbProviderObject = None
        s = self.var('db')
        if s:
            self.db = self.root.find('gws.ext.db.provider', s)
        else:
            self.db = self.root.find_first('gws.ext.db.provider')

        if not self.db:
            raise gws.Error(f'{self.uid}: db provider not found')

    def http_get_xy(self, req: gws.web.AuthRequest, p: GetXYParams) -> t.HttpResponse:
        project_uid = p.projectUid

        if project_uid:
            req.require_project(project_uid)

        alkis = self.find_first('gws.common.application').find_action('alkis', project_uid)

        if p.alkisFs:
            query = gws.ext.action.alkis.FsQueryParams({
                'alkisFs': p.alkisFs
            })
            total, features = alkis.find_fs(query, self.crs, allow_eigentuemer=False, allow_buchung=False, limit=1)

        elif p.alkisAd:
            query = gws.ext.action.alkis.FsAddressQueryParams({
                'alkisAd': p.alkisAd,
                'hausnummerNotNull': True,
            })
            total, features = alkis.find_address(query, self.crs, limit=1)
        else:
            gws.log.warn(f'gekos: bad request {req.params!r}')
            return _text('error:')

        if total == 0:
            gws.log.warn(f'gekos: not found {req.params!r}')
            return _text('error:')

        return _text_xy(features[0])

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
        'mimeType': 'text/plain',
        'content': s
    })


def _text_xy(f):
    return _text('%.3f;%.3f' % (f.attributes['x'], f.attributes['y']))
