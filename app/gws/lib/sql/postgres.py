import psycopg2
import psycopg2.extensions
import psycopg2.extras
import psycopg2.pool
import psycopg2.sql

from contextlib import contextmanager

import gws
import gws.types as t

from . import formatting


class Error(gws.Error):
    def __init__(self, sql):
        super().__init__()
        self.sql = sql


# https://www.psycopg.org/docs/usage.html#adaptation-of-python-values-to-sql-types

_type_map = {
    'array': gws.AttributeType.strlist,
    'bigint': gws.AttributeType.int,
    'bigserial': gws.AttributeType.int,
    'bit': gws.AttributeType.int,
    'bool': gws.AttributeType.bool,
    'boolean': gws.AttributeType.bool,
    'bytea': gws.AttributeType.bytes,
    'char': gws.AttributeType.str,
    'character varying': gws.AttributeType.str,
    'character': gws.AttributeType.str,
    'date': gws.AttributeType.date,
    'decimal': gws.AttributeType.float,
    'double precision': gws.AttributeType.float,
    'float4': gws.AttributeType.float,
    'float8': gws.AttributeType.float,
    'geometry': gws.AttributeType.geometry,
    'int': gws.AttributeType.int,
    'int2': gws.AttributeType.int,
    'int4': gws.AttributeType.int,
    'int8': gws.AttributeType.int,
    'integer': gws.AttributeType.int,
    'money': gws.AttributeType.float,
    'numeric': gws.AttributeType.float,
    'real': gws.AttributeType.float,
    'serial': gws.AttributeType.int,
    'serial2': gws.AttributeType.int,
    'serial4': gws.AttributeType.int,
    'serial8': gws.AttributeType.int,
    'smallint': gws.AttributeType.int,
    'smallserial': gws.AttributeType.int,
    'text': gws.AttributeType.text,
    'time': gws.AttributeType.time,
    'timestamp': gws.AttributeType.datetime,
    'timestamptz': gws.AttributeType.datetime,
    'timetz': gws.AttributeType.time,
    'varchar': gws.AttributeType.str,
}

Record = t.Dict[str, t.Any]


class Connection:
    def __init__(self, config):
        self.params = {
            'application_name': 'gws',
        }

        for p in 'host', 'port', 'user', 'password', 'database', 'connect_timeout':
            v = config.get(p)
            if v:
                self.params[p] = v

        v = config.get('timeout')
        if v:
            self.params['options'] = '-c statement_timeout=' + str(int(v) * 1000)

        self.conn = None

    def open(self):
        self.conn = psycopg2.connect(**self.params)
        self.conn.autocommit = True
        psycopg2.extensions.register_type(psycopg2.extensions.UNICODE, self.conn)
        psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY, self.conn)

    def close(self):
        try:
            self.conn.close()
        finally:
            self.conn = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _exec(self, cur, sql, params=None):
        try:
            return cur.execute(sql, params)
        except psycopg2.Error as exc:
            raise Error(sql) from exc

    #

    def format(self, sql, args=None, kwargs=None):
        return formatting.format(formatting.PostgresFormatter(), sql, args, kwargs)

    def _select(self, sql, args, kwargs):
        s, p = self.format(sql, args, kwargs)
        with self.conn.cursor() as cur:
            self._exec(cur, s, p)
            cnames = [c.name for c in cur.description]
            recs = [rec for rec in cur]
            return cnames, recs

    def select(self, sql, *args, **kwargs):
        cnames, recs = self._select(sql, args, kwargs)
        return [dict(zip(cnames, rec)) for rec in recs]

    def select_one(self, sql, *args, **kwargs):
        cnames, recs = self._select(sql, args, kwargs)
        if recs:
            return dict(zip(cnames, recs[0]))

    def select_list(self, sql, *args, **kwargs):
        cnames, recs = self._select(sql, args, kwargs)
        return [rec[0] for rec in recs]

    def select_value(self, sql, *args, **kwargs):
        cnames, recs = self._select(sql, args, kwargs)
        if recs:
            return recs[0][0]

    def count(self, table_name):
        v = self.select_value('SELECT COUNT(*) FROM {:qname}', table_name)
        return 0 if v is None else int(v)

    #

    def exec(self, sql, *args, **kwargs):
        s, p = self.format(sql, args, kwargs)
        with self.conn.cursor() as cur:
            return self._exec(cur, s, p)

    def execute(self, sql, *args, **kwargs):
        s, p = self.format(sql, args, kwargs)
        with self.conn.cursor() as cur:
            return self._exec(cur, s, p)

    #

    def insert(self, table_name, rec: Record, key_name: str = None, on_conflict: gws.Sql = None):
        text = 'INSERT INTO {:qname} ({:names}) VALUES ({:values})'
        args = [table_name, rec, rec]

        if on_conflict:
            text += ' ON CONFLICT {:sql}'
            args.append(on_conflict)

        if key_name:
            text += ' RETURNING {:name}'
            args.append(key_name)

        s, p = self.format(text, args)

        with self.conn.cursor() as cur:
            self._exec(cur, s, p)
            if key_name:
                return cur.fetchone()[0]

    def insert_many(self, table_name, recs: t.List[Record], on_conflict: gws.Sql = None, page_size: int = 100):
        if not recs:
            return

        rec_iter = iter(recs)
        keys = list(recs[0].keys())

        buf = [[], []]

        def next_chunk():
            buf[0] = []
            buf[1] = []

            for rec in rec_iter:
                buf[0].append('({:values})')
                buf[1].append([rec.get(k) for k in keys])
                if len(buf[0]) >= page_size:
                    break

            return len(buf[0]) > 0

        with self.transaction():
            while True:
                if not next_chunk():
                    break

                text = 'INSERT INTO {:qname} ({:names}) VALUES '
                args = [table_name, keys]

                text += _comma(buf[0])
                args.extend(buf[1])

                if on_conflict:
                    text += ' ON CONFLICT {:sql}'
                    args.append(on_conflict)

                s, p = self.format(text, args)
                with self.conn.cursor() as cur:
                    self._exec(cur, s, p)

    def update(self, table_name, rec: Record, key_name: str):
        key_value = None
        vals = {}

        for k, v in rec.items():
            if k == key_name:
                key_value = v
            else:
                vals[k] = v

        return self.execute('UPDATE {:qname} SET {:items} WHERE {:name} = {}', table_name, vals, key_name, key_value)

    def update_where(self, table_name, rec: Record, where: gws.Sql):
        return self.execute('UPDATE {:qname} SET {:items} WHERE {:sql}', table_name, rec, where)

    def delete(self, table_name, key_name: str, key_value):
        return self.execute('DELETE FROM {:qname} WHERE {:name} = {}', table_name, key_name, key_value)

    def delete_many(self, table_name, key_name: str, key_values: t.List[t.Any]):
        if key_values:
            return self.execute('DELETE FROM {:qname} WHERE {:name} IN ({:values})', table_name, key_name, key_values)

    def delete_where(self, table_name, where: gws.Sql):
        return self.execute('DELETE FROM {:qname} WHERE {:sql}', table_name, where)

    @contextmanager
    def transaction(self):
        with self.conn.cursor() as cur:
            self._exec(cur, 'BEGIN')
        try:
            yield self
            with self.conn.cursor() as cur:
                self._exec(cur, 'COMMIT')
        except Exception as exc:
            with self.conn.cursor() as cur:
                self._exec(cur, 'ROLLBACK')
            raise exc

    def table_names(self, schema_name):
        return self.select_list('''
            SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = {}
        ''', schema_name)

    def columns(self, table_name):
        # NB: assume postgis installed and working

        cols = []

        sql = '''
            SELECT
                a.attname,
                i.indisprimary,
                t.typname,
                postgis_typmod_type(a.atttypmod) AS gtype,
                postgis_typmod_dims(a.atttypmod) AS geom_dims,
                postgis_typmod_srid(a.atttypmod) AS geom_srid
            FROM
                pg_attribute AS a
                INNER JOIN pg_type AS t 
                    ON a.atttypid = t.oid
                LEFT JOIN pg_index AS i
                    ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE
                a.attrelid = '{:qname}'::regclass
                AND a.attnum > 0
                AND a.atttypid > 0
            ORDER BY
                a.attnum
        '''

        for r in self.select(sql, table_name):
            col = gws.SqlColumn(
                name=r['attname'],
                is_key=r['indisprimary'],
                native_type=r['typname'],
                gtype=None,
                srid=None,
                is_geometry=False,
            )
            col.type = _type_map.get(col.native_type, gws.AttributeType.str)
            if col.native_type == 'geometry':
                col.gtype = r['gtype'].upper()
                col.srid = r['geom_srid']
                col.is_geometry = True

            cols.append(col)

        return cols

    def schema_and_table(self, table_name):
        if '.' in table_name:
            return table_name.split('.', 1)
        return 'public', table_name

    def user_can(self, privilege, table_name):
        schema, tab = self.schema_and_table(table_name)
        return self.select_value('''
                SELECT COUNT(*) FROM information_schema.role_table_grants
                    WHERE
                        table_schema = {}
                        AND table_name = {}
                        AND grantee = {}
                        AND privilege_type = {}
            ''', schema, tab, self.params['user'], privilege)


_comma = ','.join
