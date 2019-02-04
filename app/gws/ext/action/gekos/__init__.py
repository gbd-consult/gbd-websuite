import time

import gws
import gws.auth.api
import gws.web
import gws.config
import gws.tools.net
import gws.gis.feature
import gws.gis.layer
import gws.gis.proj

import gws.types as t

from . import request


class PositionConfig(t.Config):
    offsetX: int
    offsetY: int
    distance: int = 0
    angle: int = 0


class Config(t.WithTypeAndAccess):
    """gekos action"""

    url: t.url
    crs: t.crsref = ''
    params: dict
    instances: t.Optional[t.List[str]]
    db: t.Optional[str]  #: database provider uid
    table: t.SqlTableConfig  #: sql table configuration
    position: t.Optional[PositionConfig]  #: position correction for points


class Object(gws.Object):
    def configure(self):
        super().configure()
        self.crs = self.var('crs', parent=True)

        self.db: t.DbProviderObject = None
        s = self.var('db')
        if s:
            self.db = self.root.find('gws.ext.db.provider', s)
        else:
            self.db = self.root.find_first('gws.ext.db.provider')

        if not self.db:
            raise gws.Error(f'{self.uid}: db provider not found')

    def http_get_xy_from_fs(self, req, p) -> t.HttpResponse:
        # ?cmd=gekosHttpGetXyFromFs&gemarkungUid=<gem>&flurnummer=<flur>&zaehler=<zaehler>&nenner=<nenner>&flurstuecksfolge=<folge>
        # httpResponse mode, s. spec page 19

        query = gws.pick(req.params, [
            'gemarkungUid',
            'flurnummer',
            'zaehler',
            'nenner',
            'flurstuecksfolge',
        ])

        # '0' values should be NULLs
        query = {k: None if v == '0' else v for k, v in query.items()}

        gws.p('http_get_xy_from_fs', req.params)

        if not query:
            gws.log.warn('gekos: bad request')
            return _text('error:')

        alkis = self.root.find_first('gws.ext.action.alkis')
        total, features = alkis.find_fs(t.Data(query), self.crs, limit=1)

        if total == 0:
            gws.log.warn('gekos: not found')
            return _text('error:')

        return _text_xy(features[0])

    def http_get_xy_from_grd(self, req, p) -> t.HttpResponse:
        # ?cmd=gekosHttpGetXyFromGrd&gemeinde=<gem>&strasse=<strasse>&hausnummer=<hausnummer>
        # httpResponse mode, s. spec page 19

        query = gws.pick(req.params, [
            'gemeinde',
            'strasse',
            'hausnummer',
        ])

        if not query:
            gws.log.warn('gekos: bad request')
            return _text('error:')

        # if no hnr is given, only select locations that have one
        if 'hausnummer' not in query or query['hausnummer'] == '0':
            query['hausnummer'] = '*'

        alkis = self.root.find_first('gws.ext.action.alkis')
        total, features = alkis.find_address(t.Data(query), self.crs, limit=1)

        if total == 0:
            gws.log.warn('gekos: not found')
            return _text('error:')

        return _text_xy(features[0])

    def load_data(self):
        """Load gekos data into a postgis table."""

        # @TODO typing, the db provider here is specifically postgis
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
            req = request.GekosRequest(options, instance, cache_lifetime=0)
            fs = req.run()
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
