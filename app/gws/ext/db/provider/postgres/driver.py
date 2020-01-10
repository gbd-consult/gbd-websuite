from contextlib import contextmanager

import psycopg2
import psycopg2.extensions
import psycopg2.extras
import psycopg2.sql
import psycopg2.pool

import gws
import gws.gis.shape
import gws.types as t

# noinspection PyArgumentList
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
# noinspection PyArgumentList
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)

Error = psycopg2.Error

# http://initd.org/psycopg/docs/usage.html?highlight=smallint#adaptation-of-python-values-to-sql-types

_type_map = {
    'array': t.AttributeType.list,
    'bigint': t.AttributeType.int,
    'int8': t.AttributeType.int,
    'bigserial': t.AttributeType.int,
    'serial8': t.AttributeType.int,
    'bit': t.AttributeType.int,
    'boolean': t.AttributeType.bool,
    'bool': t.AttributeType.bool,
    'bytea': t.AttributeType.bytes,
    'character': t.AttributeType.str,
    'char': t.AttributeType.str,
    'character varying': t.AttributeType.str,
    'varchar': t.AttributeType.str,
    'date': t.AttributeType.date,
    'geometry': t.AttributeType.geometry,
    'double precision': t.AttributeType.float,
    'float8': t.AttributeType.float,
    'integer': t.AttributeType.int,
    'int': t.AttributeType.int,
    'int4': t.AttributeType.int,
    'money': t.AttributeType.float,
    'numeric': t.AttributeType.float,
    'decimal': t.AttributeType.float,
    'real': t.AttributeType.float,
    'float4': t.AttributeType.float,
    'smallint': t.AttributeType.int,
    'int2': t.AttributeType.int,
    'smallserial': t.AttributeType.int,
    'serial2': t.AttributeType.int,
    'serial': t.AttributeType.int,
    'serial4': t.AttributeType.int,
    'text': t.AttributeType.str,
    'time': t.AttributeType.time,
    'timetz': t.AttributeType.time,
    'timestamp': t.AttributeType.datetime,
    'timestamptz': t.AttributeType.datetime,
}


class Connection:
    def __init__(self, params):
        self.params = params
        self.conn = None
        self.itersize = params.get('itersize', 100)

    def __enter__(self):
        # pool_key = 'psycopg2.pool' + _dict_hash(self.params)
        # self.pool: psycopg2.pool.ThreadedConnectionPool = gws.get_global(pool_key, self._connection_pool)
        # self.conn = self.pool.getconn()
        self.conn = psycopg2.connect(**self.params)
        self.conn.autocommit = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # self.pool.putconn(self.conn)
        self.conn.close()
        return False

    def _connection_pool(self):
        gws.log.debug(f'connection pool created')
        return psycopg2.pool.ThreadedConnectionPool(1, 20, **self.params)

    def _exec(self, cur, sql, params=None):
        try:
            return cur.execute(sql, params)
        except Error:
            gws.log.debug('FAILED SQL:')
            for s in str(sql).splitlines():
                gws.log.debug(s)
            raise

    def server_select(self, sql, params=None):
        uid = 'cur' + gws.random_string(32)
        cnames = None

        self.exec('BEGIN')
        try:
            with self.conn.cursor() as cur:
                self._exec(cur, f'DECLARE {uid} CURSOR FOR {sql}', params)
                while True:
                    self._exec(cur, f'FETCH FORWARD {self.itersize} FROM {uid}')
                    rs = cur.fetchall()
                    if not rs:
                        break
                    if not cnames:
                        cnames = [c.name for c in cur.description]
                    for rec in rs:
                        yield dict(zip(cnames, rec))
        finally:
            if self.conn:
                self.exec('COMMIT')

    def select(self, sql, params=None):
        with self.conn.cursor() as cur:
            self._exec(cur, sql, params)
            cnames = [c.name for c in cur.description]
            for rec in cur:
                yield dict(zip(cnames, rec))

    def select_one(self, sql, params=None):
        rs = list(self.select(sql, params))
        return rs[0] if rs else None

    def select_list(self, sql, params=None):
        rs = self.select(sql, params)
        return [list(r.values())[0] for r in rs]

    def select_value(self, sql, params=None):
        r = self.select_one(sql, params)
        return list(r.values())[0] if r else None

    def count(self, table_name):
        return self.select_value(f'SELECT COUNT(*) FROM {self.quote_table(table_name)}')

    def exec(self, sql, params=None):
        with self.conn.cursor() as cur:
            return self._exec(cur, sql, params)

    def execute(self, sql, params=None):
        with self.conn.cursor() as cur:
            return self._exec(cur, sql, params)

    def execute_many(self, *pairs):
        with self.conn.cursor() as cur:
            for p in pairs:
                sql, params = p[0], p[1] if len(p) > 1 else None
                self._exec(cur, sql, params)

    @contextmanager
    def transaction(self):
        self.exec('BEGIN')
        try:
            yield self
            self.exec('COMMIT')
        except:
            self.exec('ROLLBACK')
            raise

    def table_names(self, schema):
        rs = self.select('''
            SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
        ''', [schema])

        return [r['table_name'] for r in rs]

    def columns(self, table_name):
        schema, tab = self.schema_and_table(table_name)

        # NB: assume postgis installed and working

        rs = self.select('''
            SELECT f_geometry_column, srid, type 
            FROM geometry_columns
            WHERE f_table_schema = %s AND f_table_name = %s 
        ''', [schema, tab])

        geom_cols = {
            r['f_geometry_column']: {
                'type': r['type'].lower(),
                'crs': 'EPSG:%s' % r['srid']
            }
            for r in rs
        }

        rs = self.select('''
            SELECT ccu.column_name AS name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
            WHERE tc.table_schema = %s AND tc.table_name = %s
        ''', [schema, tab])

        key_cols = set(r['name'] for r in rs)

        rs = self.select('''
            SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s
        ''', [schema, tab])

        cols = []

        for r in rs:
            name = r['column_name']
            col = {
                'name': name,
                'is_key': name in key_cols,
                'geom_type': None,
                'crs': None,
                'is_geometry': False,
            }
            if name in geom_cols:
                col['crs'] = geom_cols[name]['crs']
                col['type'] = t.AttributeType.geometry
                col['native_type'] = geom_cols[name]['type']
                col['geom_type'] = col['native_type']
                col['is_geometry'] = True
            else:
                col['native_type'] = (r['udt_name'] if r['data_type'].upper() == 'USER-DEFINED' else r['data_type']).lower()
                col['type'] = _type_map.get(col['native_type'], 'str')
                col['is_geometry'] = False

            cols.append(col)

        return cols

    def insert_one(self, table_name, key_column, rec: dict, with_id=False):
        fields = []
        placeholders = []
        values = []

        for k, v in rec.items():
            if v is None:
                continue
            fields.append(self.quote_ident(k))
            if isinstance(v, (list, tuple)):
                placeholders.append(v[0])
                values.extend(v[1:])
            else:
                placeholders.append('%s')
                values.append(v)

        sql = f'''
            INSERT INTO {self.quote_table(table_name)} 
            ({_comma(fields)}) 
            VALUES ({_comma(placeholders)})
        '''

        if with_id:
            sql += f"RETURNING {self.quote_ident(key_column)}"

        with self.conn.cursor() as cur:
            self._exec(cur, sql, values)
            if with_id:
                return cur.fetchone()[0]

    def update(self, table_name, key_column, rec: dict):
        values = []
        sets = []
        uid = None

        for k, v in rec.items():
            if k == key_column:
                uid = v
                continue
            if isinstance(v, (list, tuple)):
                ph = v[0]
                values.extend(v[1:])
            else:
                ph = '%s'
                values.append(v)

            sets.append(f'{self.quote_ident(k)}={ph}')

        if uid is None:
            raise Error(f'no primary key found for update')

        values.append(uid)

        sql = f'''
            UPDATE {self.quote_table(table_name)} 
            SET {_comma(sets)}
            WHERE {self.quote_ident(key_column)}=%s
        '''

        return self.exec(sql, values)

    def delete_many(self, table_name, key_column, uids):
        values = list(uids)
        if not values:
            return

        placeholders = _comma('%s' for _ in values)
        sql = f'''
            DELETE FROM {self.quote_table(table_name)} 
            WHERE {self.quote_ident(key_column)} IN ({placeholders})
        '''

        return self.exec(sql, values)

    def insert_many(self, table_name: str, recs: t.List[dict], on_conflict=None, page_size=100):
        if not recs:
            return
        all_cols = set(c['name'] for c in self.columns(table_name))

        cols = set()
        for rec in recs:
            cols.update(rec)
        cols = sorted(c for c in cols if c in all_cols)

        template = '(' + _comma('%s' for _ in cols) + ')'
        colnames = _comma(self.quote_ident(c) for c in cols)

        sql = f'INSERT INTO {self.quote_table(table_name)} ({colnames}) VALUES %s'
        if on_conflict:
            sql += f' ON CONFLICT {on_conflict}'

        def seq():
            for rec in recs:
                yield [rec.get(c) for c in cols]

        with self.conn.cursor() as cur:
            return psycopg2.extras.execute_values(cur, sql, seq(), template, page_size)

    def schema_and_table(self, table_name):
        if '.' in table_name:
            return table_name.split('.', 1)
        return 'public', table_name

    def user_can(self, privilege, table_name):
        schema, tab = self.schema_and_table(table_name)
        return self.select_value('''
            SELECT COUNT(*) FROM information_schema.role_table_grants 
                WHERE 
                    table_schema = %s
                    AND table_name = %s
                    AND grantee = %s
                    AND privilege_type = %s 
        ''', [schema, tab, self.params['user'], privilege])

    def quote_table(self, table_name, schema=None):
        s, tab = self.schema_and_table(table_name)
        return self.quote_ident(schema or s) + '.' + self.quote_ident(tab)

    def quote_ident(self, s):
        return psycopg2.extensions.quote_ident(s, self.conn)


def _comma(s):
    return ','.join(s)


def _dict_hash(d):
    s = ''
    for k, v in sorted(d.items()):
        s += f'{k}={v} '
    return s


def _chunked(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf
