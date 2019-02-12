from contextlib import contextmanager

import psycopg2
import psycopg2.extensions
import psycopg2.extras
import psycopg2.sql
import psycopg2.pool

import gws
import gws.gis.shape

# noinspection PyArgumentList
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
# noinspection PyArgumentList
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)

Error = psycopg2.Error


class Connection:
    def __init__(self, params):
        self.params = params
        self.conn = None
        self.itersize = params.get('itersize', 100)

    def __enter__(self):
        pool_key = 'psycopg2.pool' + _dict_hash(self.params)
        self.pool: psycopg2.pool.ThreadedConnectionPool = gws.get_global(pool_key, self._connection_pool)
        self.conn = self.pool.getconn()
        self.conn.autocommit = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.putconn(self.conn)
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

    def count(self, table):
        return self.select_value(f'SELECT COUNT(*) FROM {self.quote_table(table)}')

    def exec(self, sql, params=None):
        with self.conn.cursor() as cur:
            return self._exec(cur, sql, params)

    @contextmanager
    def transaction(self):
        self.exec('BEGIN')
        try:
            yield self
            self.exec('COMMIT')
        except:
            self.exec('ROLLBACK')
            raise

    def crs_for_column(self, table, column):
        schema, table = self.schema_and_table(table)
        r = self.select_one(
            'SELECT Find_SRID(%s, %s, %s) AS n',
            [schema, table, column])
        return 'EPSG:%s' % r['n']

    def geometry_type_for_column(self, table, column):
        schema, table = self.schema_and_table(table)
        r = self.select_one('''
            SELECT type 
            FROM geometry_columns 
            WHERE 
                f_table_schema = %s
                AND f_table_name = %s 
                AND f_geometry_column = %s    
        ''', [schema, table, column])
        return str(r['type']).upper()

    def table_names(self, schema):
        rs = self.select('''
            SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
        ''', [schema])

        return [r['table_name'] for r in rs]

    def columns(self, table):
        schema, table = self.schema_and_table(table)
        rs = self.select('''
            SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s
        ''', [schema, table])

        return {r['column_name']: r['data_type'] for r in rs}

    def insert_one(self, table, key_column, data):
        fields = []
        placeholders = []
        values = []

        for k, v in data.items():
            fields.append(self.quote_ident(k))
            if isinstance(v, (list, tuple)):
                placeholders.append(v[0])
                values.extend(v[1:])
            else:
                placeholders.append('%s')
                values.append(v)

        sql = f'''
            INSERT INTO {self.quote_table(table)} 
            ({_comma(fields)}) 
            VALUES ({_comma(placeholders)})
            RETURNING {self.quote_ident(key_column)}
        '''

        with self.conn.cursor() as cur:
            self._exec(cur, sql, values)
            return cur.fetchone()[0]

    def update(self, table, key_column, key_value, data):
        values = []
        sets = []

        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                ph = v[0]
                values.extend(v[1:])
            else:
                ph = '%s'
                values.append(v)

            sets.append(f'{self.quote_ident(k)}={ph}')

        values.append(key_value)

        sql = f'''
            UPDATE {self.quote_table(table)} 
            SET {_comma(sets)}
            WHERE {self.quote_ident(key_column)}=%s
        '''

        return self.exec(sql, values)

    def delete_many(self, table, key_column, key_values):
        values = list(key_values)
        if not values:
            return

        placeholders = _comma('%s' for _ in values)
        sql = f'''
            DELETE FROM {self.quote_table(table)} 
            WHERE {self.quote_ident(key_column)} IN ({placeholders})
        '''

        return self.exec(sql, values)

    def batch_insert(self, table, data, on_conflict=None, page_size=100):
        all_cols = self.columns(table)
        cols = sorted(col for col in data[0] if col in all_cols)
        template = '(' + _comma(f'%({col})s' for col in cols) + ')'
        colnames = _comma(self.quote_ident(s) for s in cols)

        sql = f'INSERT INTO {self.quote_table(table)} ({colnames}) VALUES %s'
        if on_conflict:
            sql += f' ON CONFLICT {on_conflict}'

        with self.conn.cursor() as cur:
            return psycopg2.extras.execute_values(cur, sql, data, template, page_size)

    def schema_and_table(self, table):
        if '.' in table:
            return table.split('.', 1)
        return 'public', table

    def user_can(self, priv, table):
        s, tab = self.schema_and_table(table)
        return self.select_value('''
            SELECT COUNT(*) FROM information_schema.role_table_grants 
                WHERE 
                    table_schema = %s
                    AND table_name = %s
                    AND grantee = %s
                    AND privilege_type = %s 
        ''', [s, tab, self.params['user'], priv])

    def quote_table(self, table, schema=None):
        s, tab = self.schema_and_table(table)
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
