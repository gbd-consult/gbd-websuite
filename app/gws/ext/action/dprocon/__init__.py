"""Intergration with the DProCon software."""

import re
import time
import os

import gws
import gws.common.action
import gws.common.db
import gws.common.layer
import gws.common.template
import gws.ext.helper.alkis
import gws.gis.feature
import gws.gis.proj
import gws.gis.shape
import gws.tools.date
import gws.tools.net
import gws.web.error

import gws.types as t

_INDEX_TABLE_NAME = 'dprocon_house'
_LOG_TABLE_NAME = 'dprocon_log'

_cwd = os.path.dirname(__file__)

_DEFAULT_TEMPLATES = [
    t.Config(
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


class ExtraColumnConfig:
    name: str
    type: str
    sql: str


class Config(t.WithTypeAndAccess):
    """D-Procon connector action"""

    cacheTime: t.Duration = '24h'  #: request cache life time
    dataTableName: str  #: table to store consolidated results
    dataTablePattern: str  #: pattern for result tables to consolidate
    gemeindeFilter: t.Optional[t.List[str]]  #: gemeinde (AU) ids to keep in the index
    infoTitle: str = ''  #: information box title
    requestTableName: str  #: table to store outgoing requests
    requestUrl: t.Url  #: main program url, with the placholder {REQUEST_ID}
    templates: t.Optional[t.List[t.ext.template.Config]]  #: feature formatting templates
    extraColumns: t.Optional[t.List[ExtraColumnConfig]]


class ConnectParams(t.Params):
    shapes: t.List[t.ShapeProps]


class ConnectResponse(t.Response):
    url: str


class GetDataParams(t.Params):
    requestId: str


class GetDataResponse(t.Data):
    feature: t.FeatureProps


class Object(gws.common.action.Object):
    def configure(self):
        super().configure()

        self.au_filter: t.List[str] = self.var('gemeindeFilter', default=[])
        self.data_table_name: str = self.var('dataTableName')
        self.request_table_name: str = self.var('requestTableName')
        self.request_url: str = self.var('requestUrl')
        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'), _DEFAULT_TEMPLATES)

        self.alkis: gws.ext.helper.alkis.Object = t.cast(
            gws.ext.helper.alkis.Object,
            self.root.find_first('gws.ext.helper.alkis'))
        if not self.alkis or not self.alkis.has_index:
            gws.log.warn('dprocon cannot init, no alkis index found')
            return

        self.extra_columns = self.var('extraColumns', default=[])
        self.data_fields = dict(self._std_data_fields)
        for ec in self.extra_columns:
            self.data_fields[ec.name] = ec.type

    def api_connect(self, req: t.IRequest, p: ConnectParams) -> ConnectResponse:
        req.require_project(p.projectUid)

        shape = gws.gis.shape.union(gws.gis.shape.from_props(s) for s in p.shapes)

        request_id = self._new_request(shape)
        if not request_id:
            raise gws.web.error.NotFound()

        url = self.request_url.replace('{REQUEST_ID}', request_id)

        return ConnectResponse(url=url)

    def api_get_data(self, req: t.IRequest, p: GetDataParams) -> GetDataResponse:

        req.require_project(p.projectUid)
        request_id = p.requestId
        geom = self._selection_for_request(request_id)

        if not geom:
            gws.log.warn(f'request {request_id!r} not found')
            raise gws.web.error.NotFound()

        self._populate_data_table()
        atts = self._select_data(request_id)
        shape = gws.gis.shape.from_wkb_hex(geom, self.alkis.crs)

        f = gws.gis.feature.Feature(
            uid=f'dprocon_{request_id}',
            attributes=atts,
            shape=shape,
        )

        f.apply_templates(self.templates, {'title': self.var('infoTitle')})

        return GetDataResponse(feature=f.props)

    _std_data_fields = {
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

    def sync_columns(self, conn, table_name, ignore):
        schema, name = table_name.split('.')
        rs = conn.select('''
            SELECT column_name
            FROM information_schema.columns 
            WHERE table_schema = %s and table_name = %s
        ''', [schema, name])
        cols = [list(r.values())[0] for r in rs]

        if not cols:
            return []

        sql = []

        for col, type in self.data_fields.items():
            if col not in ignore and col not in cols:
                sql.append(f'''
                    ALTER TABLE {conn.quote_table(table_name)}
                    ADD COLUMN {col} {type}
                ''')

        for col in cols:
            if col not in ignore and col not in self.data_fields:
                sql.append(f'''
                    ALTER TABLE {conn.quote_table(table_name)}
                    DROP COLUMN {col}
                ''')

        return sql

    def setup(self, remove_request_table=False):
        with self.alkis.db.connect() as conn:
            data_fields = ','.join(k + ' ' + v for k, v in self.data_fields.items())
            index_table_name = self.alkis.index_schema + '.' + _INDEX_TABLE_NAME
            alkis_schema = self.alkis.data_schema
            srid = self.alkis.crs.split(':')[1]

            au_filter = ''
            if self.au_filter:
                s = ','.join(repr(s) for s in self.au_filter)
                au_filter = f' AND h.gemeinde IN ({s})'

            sql = [
                f'''
                    CREATE TABLE IF NOT EXISTS {conn.quote_table(index_table_name)} (
                        gml_id CHARACTER(16) PRIMARY KEY,
                        {data_fields},
                        geom geometry(POINT, {srid})
                    )
                ''',

                *self.sync_columns(
                    conn, index_table_name, ignore=['gml_id', 'geom']),

                f'''
                    TRUNCATE TABLE {conn.quote_table(index_table_name)}
                ''',

                f'''
                    INSERT INTO {conn.quote_table(index_table_name)}
                        (
                            gml_id,
                            meso_key,
                            land,
                            regierungsbezirk,
                            kreis,
                            gemeinde,
                            lage,
                            hausnummer,
                            adresse,
                            bezeichnung,
                            wert,
                            beschreibung,
                            x,
                            y,
                            geom                        
                        )
                        SELECT
                            h.gml_id,
                            h.lage || ' ' || h.hausnummer,
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
                            ST_X(p.wkb_geometry),
                            ST_Y(p.wkb_geometry),
                            p.wkb_geometry
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
                    CREATE INDEX IF NOT EXISTS geom_index 
                    ON {conn.quote_table(index_table_name)} 
                    USING GIST(geom)
                ''',
            ]

            with conn.transaction():
                for s in sql:
                    conn.exec(s)
                for ec in self.extra_columns:
                    conn.exec(ec.sql)

            cnt = conn.select_value(f'SELECT COUNT(*) FROM {index_table_name}')
            print('index ok, ', cnt, 'entries')

    def _new_request(self, shape):
        self._prepare_request_table()

        features = self.alkis.db.select(t.SelectArgs({
            'shape': shape,
            'table': self.alkis.db.configure_table(t.Data(
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
                if a.name in self.data_fields
            }
            d['request_id'] = request_id
            data.append(d)

        with self.alkis.db.connect() as conn:
            conn.insert_many(self.request_table_name, data, page_size=2000)

        with self.alkis.db.connect() as conn:
            conn.exec(f'''
                UPDATE {conn.quote_table(self.request_table_name)} SET
                    selection=%s,
                    ts=%s
                WHERE
                    request_id=%s
            ''', [
                shape.transformed_to(self.alkis.crs).ewkb_hex,
                gws.tools.date.now(),
                request_id
            ])

        return request_id

    def _prepare_request_table(self):
        srid = self.alkis.crs.split(':')[1]

        with self.alkis.db.connect() as conn:
            data_fields = ','.join(k + ' ' + v for k, v in self.data_fields.items())

            sql = [
                f'''
                    CREATE TABLE IF NOT EXISTS {conn.quote_table(self.request_table_name)} (
                        id SERIAL PRIMARY KEY,
                        request_id CHARACTER VARYING,
                        {data_fields},
                        selection geometry(GEOMETRY, {srid}),
                        ts TIMESTAMP WITH TIME ZONE
                    )
                ''',
                *self.sync_columns(
                    conn, self.request_table_name, ignore=['id', 'request_id', 'selection', 'ts'])
            ]

            for s in sql:
                conn.exec(s)

            conn.exec(f'''
                    DELETE FROM {conn.quote_table(self.request_table_name)}
                    WHERE ts < CURRENT_DATE - INTERVAL '%s seconds'
                ''', [self.var('cacheTime')])

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

        exclude_fields = list(self.data_fields) + self._exclude_fields
        data = []

        with self.alkis.db.connect() as conn:
            data_schema, data_name = conn.schema_and_table(self.data_table_name)

            for tab in conn.table_names(data_schema):
                if not re.search(self.var('dataTablePattern'), tab):
                    continue

                cols = ','.join(
                    col['name']
                    for col in conn.columns(data_schema + '.' + tab)
                    if col['name'] not in exclude_fields)

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

            conn.exec(f'''
                CREATE TABLE IF NOT EXISTS {conn.quote_table(self.data_table_name)} (
                    table_name CHARACTER VARYING,
                    request_id CHARACTER VARYING,
                    field CHARACTER VARYING,
                    value BIGINT)
            ''')

            conn.exec(f'''
                TRUNCATE TABLE {conn.quote_table(self.data_table_name)}
            ''')

            if data:
                conn.insert_many(self.data_table_name, data)

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
