import re

import gws.ext.db.provider.postgres.driver
from ..data import version


class AlkisConnection(gws.ext.db.provider.postgres.driver.Connection):
    def __init__(self, params, index_schema, data_schema, crs, exclude_gemarkung):
        super().__init__(params)
        self.index_schema = index_schema
        self.data_schema = data_schema
        self.crs = crs
        self.srid = int(self.crs.split(':')[1])
        self.exclude_gemarkung = exclude_gemarkung or []

    def index_table_version(self, table):
        if table not in self.table_names(self.index_schema):
            return 0
        s = self.select_value(f"SELECT obj_description('{self.index_schema}.{table}'::regclass)")
        if not s:
            return 0
        m = re.search(r'Version:(\d+)', s)
        if not m:
            return 0
        return int(m.group(1))

    def create_index_table(self, table, sql):
        self.exec(f'DROP TABLE IF EXISTS {self.index_schema}.{table}')
        self.exec(f'CREATE TABLE {self.index_schema}.{table} ({sql})')

    def mark_index_table(self, table):
        comment = 'Version:' + str(version.INDEX)
        self.exec(f'COMMENT ON TABLE {self.index_schema}.{table} IS %s', [comment])

    def create_index_index(self, table, columns, kind):
        name = (table + '_' + re.sub(r'\W+', '_', columns) + '_' + kind).lower()
        self.exec(f'DROP INDEX IF EXISTS {self.index_schema}.{name}')
        self.exec(f'CREATE INDEX {name} ON {self.index_schema}.{table} USING {kind}({columns})')

    def index_insert(self, table, data, page_size=100):
        self.insert_many(
            self.index_schema + '.' + table,
            data,
            on_conflict='DO NOTHING',
            page_size=page_size)

    def drop_all(self):
        for tab in self.table_names(self.index_schema):
            self.exec(f'DROP TABLE IF EXISTS {self.index_schema}.{tab}')

    def validate_index_geoms(self, table):
        idx = self.index_schema
        warnings = []

        with self.transaction():
            self.exec(f'UPDATE {idx}.{table} SET isvalid = ST_IsValid(geom)')

        rs = self.select(f'SELECT gml_id, ST_IsValidReason(geom) AS reason FROM {idx}.{table} WHERE NOT isvalid')
        for r in rs:
            warnings.append(f"gml_id={r['gml_id']} error={r['reason']}")

        with self.transaction():
            self.exec(f'DELETE FROM {idx}.{table} WHERE NOT isvalid')

        return warnings

    def select_from_ax(self, table_name, columns=None, conditions=None):
        sql = self.make_select_from_ax(table_name, columns, conditions)
        return self.select(sql)

    def make_select_from_ax(self, table_name, columns=None, conditions=None):
        all_cols = set(c['name'] for c in self.columns(f'{self.data_schema}.{table_name}'))

        def v3_name(c):
            # handle norbit plugin rename issues, e.g.
            # name (v2) => zeigtaufexternes_name (v3)
            for nc in all_cols:
                if re.match(r'^[a-z]+_' + c + '$', nc):
                    return nc

        def _col_name(c):
            # expression?
            if not re.match(r'^\w+$', c):
                return c
            # column exists?
            if c in all_cols:
                return c
            # renamed column?
            v3 = v3_name(c)
            if v3 in all_cols:
                return v3 + ' AS ' + c

        if not columns:
            columns = ['*']

        cols = []
        for c in columns:
            if c == '*':
                cols.extend(all_cols)
            else:
                c = _col_name(c)
                if c:
                    cols.append(c)

        if not conditions:
            conditions = {
                'endet': '?? IS NULL',
                'advstandardmodell': "'DLKM' = any(??)",
            }

        where = []

        for c, expr in conditions.items():
            c = _col_name(c)
            if c:
                where.append(expr.replace('??', c))

        sql = f'''SELECT {','.join(cols)} FROM {self.data_schema}.{table_name}'''

        if where:
            sql += f' WHERE ' + ' AND '.join('(' + w + ')' for w in where)

        return sql
