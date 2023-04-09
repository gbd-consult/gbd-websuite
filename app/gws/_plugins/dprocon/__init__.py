"""Intergration with the DProCon software."""

import os
import re
import time

import gws.ext.helper.alkis

import gws
import gws.types as t
import gws.base.api
import gws.base.db
import gws.base.layer
import gws.base.template
import gws.lib.feature
import gws.lib.proj
import gws.lib.shape
import gws.lib.date
import gws.lib.net
import gws.base.web.error

_INDEX_TABLE_NAME = 'dprocon_house'
_LOG_TABLE_NAME = 'dprocon_log'

_cwd = os.path.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.description',
        type='html',
        text='''
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
    )
]


class Config(gws.WithAccess):
    """D-Procon connector action"""

    cacheTime: gws.Duration = '24h'  #: request cache life time
    dataTableName: str  #: table to store consolidated results
    dataTablePattern: str  #: pattern for result tables to consolidate
    gemeindeFilter: t.Optional[list[str]]  #: gemeinde (AU) ids to keep in the index
    infoTitle: str = '' #: information box title
    requestTableName: str  #: table to store outgoing requests
    requestUrl: gws.Url #: main program url, with the placholder {REQUEST_ID}
    templates: t.Optional[list[gws.ext.template.Config]]  #: feature formatting templates


class ConnectParams(gws.Params):
    shapes: list[gws.ShapeProps]


class ConnectResponse(gws.Response):
    url: str


class GetDataParams(gws.Params):
    requestId: str


class GetDataResponse(gws.Data):
    feature: gws.lib.feature.Props


class Object(gws.base.api.Action):
    def configure(self):
        

        self.au_filter: list[str] = self.cfg('gemeindeFilter', default=[])
        self.data_table_name: str = self.cfg('dataTableName')
        self.request_table_name: str = self.cfg('requestTableName')
        self.request_url: str = self.cfg('requestUrl')
        self.templates: list[gws.ITemplate] = gws.base.template.bundle(self, self.cfg('templates'), _DEFAULT_TEMPLATES)

        self.alkis: gws.ext.helper.alkis.Object = t.cast(
            gws.ext.helper.alkis.Object,
            self.root.find_first('gws.ext.helper.alkis'))
        if not self.alkis or not self.alkis.has_index:
            gws.log.warning('dprocon cannot init, no alkis index found')
            return

    def api_connect(self, req: gws.IWebRequest, p: ConnectParams) -> ConnectResponse:
        req.require_project(p.projectUid)

        shape = gws.lib.shape.union(gws.lib.shape.from_props(s) for s in p.shapes)

        request_id = self._new_request(shape)
        if not request_id:
            raise gws.base.web.error.NotFound()

        url = self.request_url.replace('{REQUEST_ID}', request_id)

        return ConnectResponse(url=url)

    def api_get_data(self, req: gws.IWebRequest, p: GetDataParams) -> GetDataResponse:

        req.require_project(p.projectUid)
        request_id = p.requestId
        geom = self._selection_for_request(request_id)

        if not geom:
            gws.log.warning(f'request {request_id!r} not found')
            raise gws.base.web.error.NotFound()

        self._populate_data_table()
        atts = self._select_data(request_id)
        shape = gws.lib.shape.from_wkb_hex(geom, self.alkis.crs)

        f = gws.lib.feature.Feature(
            uid=f'dprocon_{request_id}',
            attributes=atts,
            shape=shape,
        )

        f.apply_templates(self.templates, {'title': self.cfg('infoTitle')})

        return GetDataResponse(feature=f.props)

    _data_fields = {
        'meso_key': 'CHARACTER VARYING',
        'land': 'CHARACTER VARYING',
        'regierungsbezirk': 'CHARACTER VARYING',
        'kreis': 'CHARACTER VARYING',
        'gemeinde': 'CHARACTER VARYING',
        'lage': 'CHARACTER VARYING',
        'hausnummer': 'CHARACTER VARYING',
        'adresse': 'CHARACTER VARYING',
        'bezeichnung': 'CHARACTER VARYING',
        'wert': 'INT',
        'beschreibung': 'CHARACTER VARYING',
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
        with self.alkis.db.connect() as conn:
            data_fields = ','.join(k + ' ' + v for k, v in self._data_fields.items())
            index_table_name = conn.quote_table(_INDEX_TABLE_NAME, self.alkis.index_schema)
            alkis_schema = self.alkis.data_schema
            srid = self.alkis.crs.split(':')[1]

            au_filter = ''
            if self.au_filter:
                s = ','.join(repr(s) for s in self.au_filter)
                au_filter = f' AND h.gemeinde IN ({s})'

            sql = [
                f'''
                    DROP TABLE IF EXISTS {index_table_name} CASCADE
                ''',
                f'''
                    CREATE TABLE {index_table_name} (
                        gml_id CHARACTER(16) PRIMARY KEY,
                        {data_fields},
                        geom geometry(POINT, {srid})
                    )
                ''',
                f'''
                    INSERT INTO {index_table_name}
                        SELECT 
                            h.gml_id,
                            h.lage || ' ' || h.hausnummer AS meso_key,
                            h.land,
                            h.regierungsbezirk,
                            h.kreis,
                            h.gemeinde,
                            h.lage,
                            h.hausnummer,
                            c.bezeichnung || ' ' || h.hausnummer,
                            c.bezeichnung,
                            gf.wert,
                            gf.beschreibung,
                            ST_X(p.wkb_geometry) AS x,
                            ST_Y(p.wkb_geometry) AS y,
                            p.wkb_geometry AS geom
                        FROM
                            "{alkis_schema}".ax_lagebezeichnungmithausnummer AS h,
                            "{alkis_schema}".ax_lagebezeichnungkatalogeintrag AS c,
                            "{alkis_schema}".ap_pto AS p,
                            "{alkis_schema}".ax_gebaeude AS g,
                            "{alkis_schema}".ax_gebaeudefunktion AS gf
                        WHERE
                            p.art = 'HNR'
                            AND h.gml_id = ANY (p.dientzurdarstellungvon)
                            AND c.land = h.land
                            AND c.regierungsbezirk = h.regierungsbezirk
                            AND c.kreis = h.kreis
                            AND c.gemeinde = h.gemeinde
                            AND c.lage = h.lage
                            AND h.gml_id = ANY(g.zeigtauf)
                            AND gf.wert = g.gebaeudefunktion
                            AND c.endet IS NULL
                            AND g.endet IS NULL
                            AND h.endet IS NULL
                            AND p.endet IS NULL
                            {au_filter}
                    ON CONFLICT DO NOTHING
                ''',
                f'''
                    CREATE INDEX geom_index ON {index_table_name} USING GIST(geom)
                ''',
            ]

            with conn.transaction():
                for s in sql:
                    conn.exec(s)

            cnt = conn.select_value(f'SELECT COUNT(*) FROM {index_table_name}')
            print('index ok, ', cnt, 'entries')

    def _new_request(self, shape):
        self._prepare_request_table()

        features = self.alkis.db.select(gws.SqlSelectArgs({
            'shape': shape,
            'table': self.alkis.db.configure_table(gws.Data(
                name=self.alkis.index_schema + '.' + _INDEX_TABLE_NAME,
            ))
        }))

        if not features:
            return None

        request_id = _rand_id()
        data = []

        for f in features:
            d = {
                a.name: a.value
                for a in f.attributes
                if a.name in self._data_fields
            }
            d['request_id'] = request_id
            d['selection'] = shape.transformed_to(self.alkis.crs).ewkb_hex
            d['ts'] = gws.lib.date.now()
            data.append(d)

        with self.alkis.db.connect() as conn:
            for d in data:
                conn.insert_one(self.request_table_name, 'request_id', d)

        return request_id

    def _prepare_request_table(self):
        srid = self.alkis.crs.split(':')[1]

        with self.alkis.db.connect() as conn:
            request_table_name = conn.quote_table(self.request_table_name)
            data_fields = ','.join(k + ' ' + v for k, v in self._data_fields.items())

            conn.exec(f'''
                CREATE TABLE IF NOT EXISTS {request_table_name} ( 
                    id SERIAL PRIMARY KEY,
                    request_id CHARACTER VARYING,
                    {data_fields},
                    selection geometry(GEOMETRY, {srid}) NOT NULL,
                    ts TIMESTAMP WITH TIME ZONE
                )
            ''')

            # clean up obsolete request records
            conn.exec(f'''
                DELETE FROM {request_table_name} 
                WHERE ts < CURRENT_DATE - INTERVAL '%s seconds' 
            ''', [self.cfg('cacheTime')])

    def _selection_for_request(self, request_id):
        with self.alkis.db.connect() as conn:
            return conn.select_value(f'''
                SELECT selection 
                FROM {conn.quote_table(self.request_table_name)} 
                WHERE request_id=%s
            ''', [request_id])

    def _populate_data_table(self):
        # collect data from various dprocon-tables into a single data table
        # dprocon-tables contain (request_id, common fields, specific fields)

        exclude_fields = list(self._data_fields) + self._exclude_fields
        data = []

        with self.alkis.db.connect() as conn:
            data_schema, data_name = conn.schema_and_table(self.data_table_name)

            for tab in conn.table_names(data_schema):
                if not re.search(self.cfg('dataTablePattern'), tab):
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

            tmp = f"{self.data_table_name}{_rand_id()}"

            conn.exec(f'''
                CREATE TABLE IF NOT EXISTS {conn.quote_table(tmp)} ( 
                    table_name CHARACTER VARYING,
                    request_id CHARACTER VARYING,
                    field CHARACTER VARYING,
                    value BIGINT)
            ''')

            if data:
                conn.insert_many(tmp, data)

            conn.exec(f'''DROP TABLE IF EXISTS {conn.quote_table(self.data_table_name)} CASCADE''')
            conn.exec(f'''ALTER TABLE {conn.quote_table(tmp)} RENAME TO {data_name}''')

    def _select_data(self, request_id):
        d = {}

        with self.alkis.db.connect() as conn:
            rs = conn.select(f'''
                SELECT * 
                FROM {conn.quote_table(self.data_table_name)} 
                WHERE request_id=%s
            ''', [request_id])

            for r in rs:
                f, v = r['field'], r['value']
                if isinstance(v, int):
                    d[f] = d.get(f, 0) + v

        return d


def _rand_id():
    return str(int(time.time() * 1000))
