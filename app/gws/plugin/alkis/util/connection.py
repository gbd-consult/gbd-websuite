import re

import gws.lib.sql.postgres
import gws.lib.sql.formatting
import gws.types as t

INDEX_VERSION = 8


class AlkisConnection(gws.lib.sql.postgres.Connection):
    def __init__(self, config):
        super().__init__(config)
        self.index_schema = config.get('index_schema')
        self.data_schema = config.get('data_schema')
        self.crs = config.get('crs')
        self.srid = self.crs.srid
        self.exclude_gemarkung = config.get('exclude_gemarkung', [])

    def index_table_version(self, tab):
        idx = self.index_schema
        if tab not in self.table_names(idx):
            return 0
        comment = self.select_value(f"SELECT obj_description('{idx}.{tab}'::regclass)")
        if not comment:
            return 0
        m = re.search(r'Version:(\d+)', comment)
        if not m:
            return 0
        return int(m.group(1))

    def create_index_table(self, tab, raw_sql):
        idx = self.index_schema
        self.exec(f'DROP TABLE IF EXISTS {idx}.{tab}')
        self.exec(f'CREATE TABLE {idx}.{tab} ({raw_sql})')

    def mark_index_table(self, tab):
        idx = self.index_schema
        comment = 'Version:' + str(INDEX_VERSION)
        self.exec(f'COMMENT ON TABLE {idx}.{tab} IS {{:value}}', comment)

    def create_index_index(self, tab, column: str, kind: str):
        idx = self.index_schema
        name = f'{tab}_{column}_{kind}'.lower()
        self.exec(f'DROP INDEX IF EXISTS {idx}.{name}')
        self.exec(f'CREATE INDEX {name} ON {idx}.{tab} USING {kind}({column})')

    def index_insert(self, tab, data, page_size=100):
        self.insert_many(
            [self.index_schema, tab],
            data,
            on_conflict=gws.Sql('DO NOTHING'),
            page_size=page_size)

    def drop_all_indexes(self):
        idx = self.index_schema
        for tab in self.table_names(self.index_schema):
            ver = self.index_table_version(tab)
            if ver == INDEX_VERSION:
                self.exec(f'DROP TABLE IF EXISTS {idx}.{tab}')

    def validate_index_geoms(self, tab):
        idx = self.index_schema
        warnings = []

        with self.transaction():
            self.exec(f'UPDATE {idx}.{tab} SET isvalid = ST_IsValid(geom)')

        rs = self.select(f'SELECT gml_id, ST_IsValidReason(geom) AS reason FROM {idx}.{tab} WHERE NOT isvalid')
        for r in rs:
            warnings.append(f"gml_id={r['gml_id']} error={r['reason']}")

        with self.transaction():
            self.exec(f'DELETE FROM {idx}.{tab} WHERE NOT isvalid')

        return warnings

    def select_from_ax(self, tab, columns=None, conditions=None):
        sql = self.make_select_from_ax(tab, columns, conditions)
        return self.select(sql)

    def make_select_from_ax(self, tab, columns=None, conditions=None):
        all_cols = set(col.name for col in self.columns([self.data_schema, tab]))

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

        sql = f'''SELECT {','.join(cols)} FROM {self.data_schema}.{tab}'''

        if where:
            sql += f' WHERE ' + ' AND '.join('(' + w + ')' for w in where)

        return sql
