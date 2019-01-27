import time
import os

import gws
import gws.auth.api
import gws.web
import gws.config
import gws.tools.net
import gws.tools.date
import gws.gis.feature
import gws.gis.shape
import gws.gis.layer

import gws.types as t

dprocon_index = 'dprocon_house'
dprocon_log = 'dprocon_log'

_cwd = os.path.dirname(__file__)

_DEFAULT_FORMAT = t.FormatConfig({
    'description': t.TemplateConfig({
        'type': 'html',
        'path': _cwd + '/data.cx.html'
    })
})


class Config(t.WithTypeAndAccess):
    """dprocon action"""

    db: str = ''
    requestUrl: t.url
    crs: t.crsref = ''

    schema: str
    requestTable: str
    dataTable: str

    cacheTime: t.duration = '24h'
    title: str = ''
    indexSchema: str = 'gws'  #: schema to store gws internal indexes, must be writable
    alkisSchema: str = 'public'  #: schema where ALKIS tables are stored, must be readable


class ConnectParams(t.Data):
    projectUid: str
    shapes: t.List[t.ShapeProps]


class ConnectResponse(t.Response):
    url: str


class GetDataParams(t.Data):
    projectUid: str
    requestId: str


class GetDataResponse(t.Data):
    feature: t.FeatureProps


class Object(gws.Object):
    def configure(self):
        super().configure()

        s = self.var('db')
        self.db: t.DbProviderObject = None
        if s:
            self.db = self.root.find('gws.ext.db.provider', s)
        else:
            self.db = self.root.find_first('gws.ext.db.provider')

        # @TODO find crs from alkis
        self.crs = self.var('crs', parent=True)
        self.srid = int(self.crs.split(':')[1])

        self.data_schema = self.var('schema')
        self.request_table = self.var('requestTable')
        self.data_table = self.var('dataTable')

        self.feature_format = self.create_object('gws.common.format', _DEFAULT_FORMAT)

    def api_connect(self, req, p: ConnectParams) -> ConnectResponse:
        project = req.require_project(p.projectUid)
        shape = gws.gis.shape.union(gws.gis.shape.from_props(s) for s in p.shapes)

        request_id = self._connect(shape)
        if not request_id:
            raise gws.web.error.NotFound()

        url = self.var('requestUrl')
        url = url.replace('{REQUEST_ID}', request_id)

        return ConnectResponse({
            'url': url
        })

    def api_get_data(self, req, p: GetDataParams) -> GetDataResponse:
        project = req.require_project(p.projectUid)
        request_id = p.requestId

        with self.db.connect() as conn:
            geom = conn.select_value(f'''
                SELECT selection 
                FROM {self.data_schema}.{self.request_table} 
                WHERE request_id=%s
            ''', [request_id])

        if not geom:
            gws.log.info(f'request {request_id!r} not found')
            raise gws.web.error.NotFound()

        self._populate_data_table()
        atts = self._select_data(request_id)
        shape = gws.gis.shape.from_wkb(geom, self.crs)

        f = gws.gis.feature.Feature({
            'uid': 'dprocon_%s' % request_id,
            'attributes': atts,
            'shape': shape,
        })

        f.apply_format(self.feature_format, {'title': self.var('title')})

        return GetDataResponse({
            'feature': f.props
        })

    _data_fields = {
        'meso_key': 'CHARACTER VARYING',
        'land': 'CHARACTER VARYING',
        'regierungsbezirk': 'CHARACTER VARYING',
        'kreis': 'CHARACTER VARYING',
        'gemeinde': 'CHARACTER VARYING',
        'lage': 'CHARACTER VARYING',
        'hausnummer': 'CHARACTER VARYING',
        'bezeichnung': 'CHARACTER VARYING',
        'x': 'FLOAT',
        'y': 'FLOAT',
    }

    _exclude_fields = [
        'id',
        'gid',
        'selection',
        'ts',
        'bezirk'
    ]

    def create_index(self, user, password):
        with self.db.connect({'user': user, 'password': password}) as conn:
            index_table = conn.quote_table(dprocon_index, self.var('indexSchema'))
            ds = self.var('alkisSchema')

            df = ','.join(k + ' ' + v for k, v in self._data_fields.items())

            sql = [
                f'''
                    DROP TABLE IF EXISTS {index_table} CASCADE
                ''',
                f'''
                    CREATE TABLE {index_table} (
                        gml_id CHARACTER(16) NOT NULL,
                        {df},
                        geom geometry(POINT, {self.srid})
                    )
                ''',
                f'''
                    INSERT INTO {index_table}
                        SELECT 
                            h.gml_id,
                            h.lage || ' ' || h.hausnummer AS meso_key,
                            h.land,
                            h.regierungsbezirk,
                            h.kreis,
                            h.gemeinde,
                            h.lage,
                            h.hausnummer,
                            c.bezeichnung,
                            ST_X(p.wkb_geometry) AS x,
                            ST_Y(p.wkb_geometry) AS y,
                            p.wkb_geometry AS geom
                        FROM
                            "{ds}".ax_lagebezeichnungmithausnummer AS h,
                            "{ds}".ax_lagebezeichnungkatalogeintrag AS c,
                            "{ds}".ap_pto AS p
                        WHERE
                            p.art = 'HNR'
                            AND h.gml_id = ANY (p.dientzurdarstellungvon)
                            AND p.endet IS NULL
                            AND h.endet IS NULL
                            AND c.land = h.land
                            AND c.regierungsbezirk = h.regierungsbezirk
                            AND c.kreis = h.kreis
                            AND c.gemeinde = h.gemeinde
                            AND c.lage = h.lage
                ''',
                f'''
                    CREATE INDEX geom_index ON {index_table} USING GIST(geom)
                '''
            ]

            with conn.transaction():
                for s in sql:
                    conn.exec(s)

            cnt = conn.select_value(f'SELECT COUNT(*) FROM {index_table}')
            print('index ok, ', cnt, 'entries')

    def create_tables(self, user, password):
        with self.db.connect({'user': user, 'password': password}) as conn:
            request_table = self.data_schema + '.' + self.request_table

            df = ','.join(k + ' ' + v for k, v in self._data_fields.items())

            sql = [
                f'''
                    DROP TABLE IF EXISTS {request_table} CASCADE
                ''',
                f'''
                    CREATE TABLE IF NOT EXISTS {request_table} ( 
                        id SERIAL PRIMARY KEY,
                        request_id CHARACTER VARYING,
                        
                        {df},
                        
                        selection geometry(GEOMETRY,{self.srid}) NOT NULL,
                        ts TIMESTAMP WITH TIME ZONE
                    )
                '''
            ]

            with conn.transaction():
                for s in sql:
                    conn.exec(s)

            print('setup ok')

    def _connect(self, shape):
        features = self.db.select(t.SelectArgs({
            'shape': shape,
            'table': t.SqlTableConfig({
                'name': self.var('indexSchema') + '.' + dprocon_index,
                'geometryColumn': 'geom'
            })
        }))

        if not features:
            return None

        request_id = _rand_id()
        ts = gws.tools.date.now()

        data = []

        for f in features:
            d = {k: f.attributes.get(k) for k in self._data_fields}
            d['request_id'] = request_id
            d['selection'] = ['ST_SetSRID(%s::geometry,%s)', shape.wkb_hex, self.srid]
            d['ts'] = ts
            data.append(d)

        secs = self.var('cacheTime')
        request_table = self.data_schema + '.' + self.request_table

        with self.db.connect() as conn:
            with conn.transaction():
                conn.exec(f'''
                    DELETE 
                    FROM {request_table} 
                    WHERE ts < CURRENT_DATE - INTERVAL '{secs} seconds' 
                ''')
                for d in data:
                    conn.insert_one(request_table, 'id', d)

        # tab = log_table()
        # db.run(_f('''INSERT INTO {tab}(request_id,selection,ts)
        #     VALUES(%s,%s,%s)
        # '''), [request_id, json.dumps(wkts), util.now()])

        return request_id

    def _populate_data_table(self):
        # collect data from various dprocon-tables into a single data table
        # dprocon-tables contain (request_id, common fields, specific fields)

        exclude_fields = list(self._data_fields) + self._exclude_fields
        data = []

        with self.db.connect() as conn:

            rs = conn.select(f'''
                SELECT table_name 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE column_name='request_id' AND table_schema=%s
            ''', [self.data_schema])

            tables = [r['table_name'] for r in rs]

            for tab in tables:
                if tab in (self.request_table, self.data_table):
                    continue

                cols = ','.join(
                    c
                    for c in conn.columns(self.data_schema + '.' + tab)
                    if c not in exclude_fields)

                if not cols:
                    continue

                sql = f'''
                    SELECT request_id, {cols} 
                    FROM {self.data_schema}.{tab}
                    WHERE request_id::text != '0'    
                '''

                for r in conn.server_select(sql):
                    for f, v in r.items():
                        if isinstance(v, int):
                            data.append({
                                'table_name': tab,
                                'request_id': r['request_id'],
                                'field': f,
                                'value': v
                            })

            tmp_data_table = self.data_schema + '.' + self.data_table + str(_rand_id())

            conn.exec(f'''
                CREATE TABLE IF NOT EXISTS {tmp_data_table} ( 
                    table_name CHARACTER VARYING,
                    request_id CHARACTER VARYING,
                    field CHARACTER VARYING,
                    value BIGINT
                )
            ''')

            if data:
                conn.batch_insert(tmp_data_table, data)

            dt = self.data_schema + '.' + self.data_table
            conn.exec(f'''DROP TABLE IF EXISTS {dt} CASCADE''')
            conn.exec(f'''ALTER TABLE {tmp_data_table} RENAME TO {self.data_table}''')

    def _select_data(self, request_id):
        dt = self.data_schema + '.' + self.data_table
        d = {}

        with self.db.connect() as conn:
            for r in conn.select(f'''SELECT * FROM {dt} WHERE request_id=%s''', [request_id]):
                f, v = r['field'], r['value']
                if isinstance(v, int):
                    d[f] = d.get(f, 0) + v

        return d


def _rand_id():
    return str(int(time.time() * 1000))
