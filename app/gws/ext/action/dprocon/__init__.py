import re
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
import gws.common.layer
import gws.gis.proj

import gws.types as t

_INDEX_TABLE_NAME = 'dprocon_house'
_LOG_TABLE_NAME = 'dprocon_log'

_cwd = os.path.dirname(__file__)

_DEFAULT_FORMAT = t.FeatureFormatConfig({
    'description': t.TemplateConfig({
        'type': 'html',
        'text': '''
            <p class="head">{title}</p>
            
            <table>
            @each attributes as k, v
                <tr>
                    <td>{k | html}</td>
                    <td align=right>{v | html}</td>
                </tr>
            @end
            </table>
        '''
    })
})


class Config(t.WithTypeAndAccess):
    """D-Procon action"""

    db: str = ''
    requestUrl: t.Url
    crs: t.Crs = ''

    requestTable: t.SqlTableConfig  #: table to store outgoing requests
    dataTable: t.SqlTableConfig  #: table to store consolidated results
    dataTablePattern: str  #: pattern for result tables to consolidate

    cacheTime: t.Duration = '24h'
    infoTitle: str = ''

    indexSchema: str = 'gws'  #: schema to store gws internal indexes, must be writable
    alkisSchema: str = 'public'  #: schema where ALKIS tables are stored, must be readable


class ConnectParams(t.Params):
    shapes: t.List[t.ShapeProps]


class ConnectResponse(t.Response):
    url: str


class GetDataParams(t.Params):
    requestId: str


class GetDataResponse(t.Data):
    feature: t.FeatureProps


class Object(gws.ActionObject):
    def __init__(self):
        super().__init__()
        self.db: t.SqlProviderObject = None
        
    def configure(self):
        super().configure()

        s = self.var('db')
        if s:
            self.db = self.root.find('gws.ext.db.provider', s)
        else:
            self.db = self.root.find_first('gws.ext.db.provider')

        # @TODO find crs from alkis
        self.crs = self.var('crs')
        self.srid = gws.gis.proj.as_srid(self.crs)

        self.request_table = self.var('requestTable')
        self.data_table = self.var('dataTable')

        self.feature_format = self.create_object('gws.common.format', _DEFAULT_FORMAT)

    def api_connect(self, req: t.WebRequest, p: ConnectParams) -> ConnectResponse:
        req.require_project(p.projectUid)

        shape = gws.gis.shape.union(gws.gis.shape.from_props(s) for s in p.shapes)

        request_id = self._new_request(shape)
        if not request_id:
            raise gws.web.error.NotFound()

        url = self.var('requestUrl')
        url = url.replace('{REQUEST_ID}', request_id)

        return ConnectResponse({
            'url': url
        })

    def api_get_data(self, req: t.WebRequest, p: GetDataParams) -> GetDataResponse:

        req.require_project(p.projectUid)
        request_id = p.requestId
        geom = self._selection_for_request(request_id)

        if not geom:
            gws.log.info(f'request {request_id!r} not found')
            raise gws.web.error.NotFound()

        self._populate_data_table()
        atts = self._select_data(request_id)
        shape = gws.gis.shape.from_wkb(geom, self.crs)

        f = gws.gis.feature.new({
            'uid': 'dprocon_%s' % request_id,
            'attributes': atts,
            'shape': shape,
        })

        f.apply_format(self.feature_format, {'title': self.var('infoTitle')})

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

    def setup(self):
        with self.db.connect() as conn:
            request_table = conn.quote_table(self.request_table.name)
            data_fields = ','.join(k + ' ' + v for k, v in self._data_fields.items())
            index_table = conn.quote_table(_INDEX_TABLE_NAME, self.var('indexSchema'))
            alkis_schema = self.var('alkisSchema')

            sql = [
                f'''
                    DROP TABLE IF EXISTS {index_table} CASCADE
                ''',
                f'''
                    CREATE TABLE {index_table} (
                        gml_id CHARACTER(16) NOT NULL,
                        {data_fields},
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
                            "{alkis_schema}".ax_lagebezeichnungmithausnummer AS h,
                            "{alkis_schema}".ax_lagebezeichnungkatalogeintrag AS c,
                            "{alkis_schema}".ap_pto AS p
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
                ''',
            ]

            with conn.transaction():
                for s in sql:
                    conn.exec(s)

            cnt = conn.select_value(f'SELECT COUNT(*) FROM {index_table}')
            print('index ok, ', cnt, 'entries')

    def _new_request(self, shape):
        self._prepare_request_table()

        features = self.db.select(t.SelectArgs({
            'shape': shape,
            'table': t.SqlTableConfig({
                'name': self.var('indexSchema') + '.' + _INDEX_TABLE_NAME,
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

        with self.db.connect() as conn:
            for d in data:
                conn.insert_one(self.request_table.name, 'request_id', d)

        # tab = log_table()
        # db.run(_f('''INSERT INTO {tab}(request_id,selection,ts)
        #     VALUES(%s,%s,%s)
        # '''), [request_id, json.dumps(wkts), util.now()])

        return request_id

    def _prepare_request_table(self):
        with self.db.connect() as conn:
            request_table = conn.quote_table(self.request_table.name)
            data_fields = ','.join(k + ' ' + v for k, v in self._data_fields.items())

            # ensure the requests table
            conn.exec(f'''
                CREATE TABLE IF NOT EXISTS {request_table} ( 
                    id SERIAL PRIMARY KEY,
                    request_id CHARACTER VARYING,
                    {data_fields},
                    selection geometry(GEOMETRY, {self.srid}) NOT NULL,
                    ts TIMESTAMP WITH TIME ZONE
                )
            ''')

            # clean up obsolete request records
            conn.exec(f'''
                DELETE FROM {request_table} 
                WHERE ts < CURRENT_DATE - INTERVAL '%s seconds' 
            ''', [self.var('cacheTime')])

    def _selection_for_request(self, request_id):
        with self.db.connect() as conn:
            tab = conn.quote_table(self.request_table.name)
            return conn.select_value(f'''
                SELECT selection 
                FROM {tab} 
                WHERE request_id=%s
            ''', [request_id])

    def _populate_data_table(self):
        # collect data from various dprocon-tables into a single data table
        # dprocon-tables contain (request_id, common fields, specific fields)

        exclude_fields = list(self._data_fields) + self._exclude_fields
        data = []

        with self.db.connect() as conn:
            data_schema, data_name = conn.schema_and_table(self.data_table.name)

            for tab in conn.table_names(data_schema):
                if not re.search(self.var('dataTablePattern'), tab):
                    continue

                cols = ','.join(
                    c
                    for c in conn.columns(data_schema + '.' + tab)
                    if c not in exclude_fields)

                if not cols:
                    continue

                sql = f'''
                    SELECT request_id, {cols}
                    FROM {data_schema}.{tab}
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

            tmp = self.data_table.name + str(_rand_id())

            conn.exec(f'''
                CREATE TABLE IF NOT EXISTS {conn.quote_table(tmp)} ( 
                    table_name CHARACTER VARYING,
                    request_id CHARACTER VARYING,
                    field CHARACTER VARYING,
                    value BIGINT)
            ''')

            if data:
                conn.batch_insert(tmp, data)

            conn.exec(f'''DROP TABLE IF EXISTS {conn.quote_table(self.data_table.name)} CASCADE''')
            conn.exec(f'''ALTER TABLE {conn.quote_table(tmp)} RENAME TO {data_name}''')

    def _select_data(self, request_id):
        d = {}

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT * 
                FROM {conn.quote_table(self.data_table.name)} 
                WHERE request_id=%s
            ''', [request_id])

            for r in rs:
                f, v = r['field'], r['value']
                if isinstance(v, int):
                    d[f] = d.get(f, 0) + v

        return d


def _rand_id():
    return str(int(time.time() * 1000))
