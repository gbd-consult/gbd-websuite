from typing import Optional, cast
import contextlib

import gws
import gws.lib.sa as sa


class Config(gws.Config):
    """Database provider"""

    schemaCacheLifeTime: gws.Duration = 3600
    """life time for schema caches"""


class Object(gws.DatabaseProvider):
    saEngine: sa.Engine
    saMetaMap: dict[str, sa.MetaData]
    saConnection: Optional[sa.Connection]
    saConnectionCount: int

    def __getstate__(self):
        return gws.u.omit(vars(self), 'saMetaMap', 'saEngine', 'saConnection')

    def configure(self):
        self.url = ''
        self.saEngine = self.engine(poolclass=sa.NullPool)
        self.saMetaMap = {}
        self.saConnection = None
        self.saConnectionCount = 0

    def activate(self):
        self.saEngine = self.engine()
        self.saMetaMap = {}
        self.saConnection = None
        self.saConnectionCount = 0

    def reflect_schema(self, schema: str):
        if schema in self.saMetaMap:
            return

        def _load():
            md = sa.MetaData(schema=schema)

            # introspecting the whole schema is generally faster
            # but what if we only need a single table from a big schema?
            # @TODO add options for reflection

            gws.debug.time_start(f'AUTOLOAD {self.uid=} {schema=}')
            with self.connect() as conn:
                md.reflect(conn, schema, resolve_fks=False, views=True)
            gws.debug.time_end()

            return md

        life_time = self.cfg('schemaCacheLifeTime', 0)
        if not life_time:
            self.saMetaMap[schema] = _load()
        else:
            self.saMetaMap[schema] = gws.u.get_cached_object(f'database_metadata_schema_{schema}', life_time, _load)

    @contextlib.contextmanager
    def connect(self):
        if self.saConnection is None:
            self.saConnection = self.saEngine.connect()
            # gws.log.debug(f'db connection opened: {self.saConnection}')
            self.saConnectionCount = 1
        else:
            self.saConnectionCount += 1

        try:
            yield self.saConnection
        finally:
            self.saConnectionCount -= 1
            if self.saConnectionCount == 0:
                self.saConnection.close()
                # gws.log.debug(f'db connection closed: {self.saConnection}')
                self.saConnection = None

    def table(self, table_name, **kwargs):
        sa_table = self._sa_table(table_name)
        if sa_table is None:
            raise sa.Error(f'table {table_name!r} not found')
        return sa_table

    def count(self, table):
        sa_table = self._sa_table(table)
        if sa_table is None:
            return 0
        sql = sa.select(sa.func.count()).select_from(sa_table)
        with self.connect() as conn:
            try:
                r = list(conn.execute(sql))
                return r[0][0]
            except sa.Error:
                conn.rollback()
                return 0

    def has_table(self, table_name: str):
        sa_table = self._sa_table(table_name)
        return sa_table is not None

    def _sa_table(self, tab_or_name) -> sa.Table:
        if isinstance(tab_or_name, sa.Table):
            return tab_or_name
        schema, name = self.split_table_name(tab_or_name)
        self.reflect_schema(schema)
        # see _get_table_key in sqlalchemy/sql/schema.py
        table_key = schema + '.' + name
        return self.saMetaMap[schema].tables.get(table_key)

    def column(self, table, column_name):
        sa_table = self._sa_table(table)
        try:
            return sa_table.columns[column_name]
        except KeyError:
            raise sa.Error(f'column {str(table)}.{column_name!r} not found')

    def has_column(self, table, column_name):
        sa_table = self._sa_table(table)
        return sa_table is not None and column_name in sa_table.columns

    def select_text(self, sql, **kwargs):
        with self.connect() as conn:
            try:
                return [
                    gws.u.to_dict(r)
                    for r in conn.execute(sa.text(sql), kwargs)
                ]
            except sa.Error:
                conn.rollback()
                raise

    def execute_text(self, sql, **kwargs):
        with self.connect() as conn:
            try:
                res = conn.execute(sa.text(sql), kwargs)
                conn.commit()
                return res
            except sa.Error:
                conn.rollback()
                raise

    SA_TO_ATTR = {
        # common: sqlalchemy.sql.sqltypes

        'BIGINT': gws.AttributeType.int,
        'BOOLEAN': gws.AttributeType.bool,
        'CHAR': gws.AttributeType.str,
        'DATE': gws.AttributeType.date,
        'DOUBLE_PRECISION': gws.AttributeType.float,
        'INTEGER': gws.AttributeType.int,
        'NUMERIC': gws.AttributeType.float,
        'REAL': gws.AttributeType.float,
        'SMALLINT': gws.AttributeType.int,
        'TEXT': gws.AttributeType.str,
        # 'UUID': ...,
        'VARCHAR': gws.AttributeType.str,

        # postgres specific: sqlalchemy.dialects.postgresql.types

        # 'JSON': ...,
        # 'JSONB': ...,
        # 'BIT': ...,
        'BYTEA': gws.AttributeType.bytes,
        # 'CIDR': ...,
        # 'INET': ...,
        # 'MACADDR': ...,
        # 'MACADDR8': ...,
        # 'MONEY': ...,
        'TIME': gws.AttributeType.time,
        'TIMESTAMP': gws.AttributeType.datetime,
    }

    # @TODO proper support for Z/M geoms

    SA_TO_GEOM = {
        'POINT': gws.GeometryType.point,
        'POINTM': gws.GeometryType.point,
        'POINTZ': gws.GeometryType.point,
        'POINTZM': gws.GeometryType.point,
        'LINESTRING': gws.GeometryType.linestring,
        'LINESTRINGM': gws.GeometryType.linestring,
        'LINESTRINGZ': gws.GeometryType.linestring,
        'LINESTRINGZM': gws.GeometryType.linestring,
        'POLYGON': gws.GeometryType.polygon,
        'POLYGONM': gws.GeometryType.polygon,
        'POLYGONZ': gws.GeometryType.polygon,
        'POLYGONZM': gws.GeometryType.polygon,
        'MULTIPOINT': gws.GeometryType.multipoint,
        'MULTIPOINTM': gws.GeometryType.multipoint,
        'MULTIPOINTZ': gws.GeometryType.multipoint,
        'MULTIPOINTZM': gws.GeometryType.multipoint,
        'MULTILINESTRING': gws.GeometryType.multilinestring,
        'MULTILINESTRINGM': gws.GeometryType.multilinestring,
        'MULTILINESTRINGZ': gws.GeometryType.multilinestring,
        'MULTILINESTRINGZM': gws.GeometryType.multilinestring,
        'MULTIPOLYGON': gws.GeometryType.multipolygon,
        # 'GEOMETRYCOLLECTION': gws.GeometryType.geometrycollection,
        # 'CURVE': gws.GeometryType.curve,
    }

    UNKNOWN_TYPE = gws.AttributeType.str
    UNKNOWN_ARRAY_TYPE = gws.AttributeType.strlist

    def describe(self, table):
        sa_table = self._sa_table(table)
        if sa_table is None:
            raise sa.Error(f'table {table!r} not found')

        schema = sa_table.schema
        name = sa_table.name

        desc = gws.DataSetDescription(
            columns=[],
            columnMap={},
            fullName=self.join_table_name(schema, name),
            geometryName='',
            geometrySrid=0,
            geometryType='',
            name=name,
            schema=schema
        )

        for n, sa_col in enumerate(cast(list[sa.Column], sa_table.columns)):
            col = self.describe_column(table, sa_col.name)
            col.columnIndex = n
            desc.columns.append(col)
            desc.columnMap[col.name] = col

        for col in desc.columns:
            if col.geometryType:
                desc.geometryName = col.name
                desc.geometryType = col.geometryType
                desc.geometrySrid = col.geometrySrid
                break

        return desc

    def describe_column(self, table, column_name) -> gws.ColumnDescription:
        sa_col = self.column(table, column_name)

        col = gws.ColumnDescription(
            columnIndex=0,
            comment=str(sa_col.comment or ''),
            default=sa_col.default,
            geometrySrid=0,
            geometryType='',
            isAutoincrement=bool(sa_col.autoincrement),
            isNullable=bool(sa_col.nullable),
            isPrimaryKey=bool(sa_col.primary_key),
            isUnique=bool(sa_col.unique),
            hasDefault=sa_col.server_default is not None,
            name=str(sa_col.name),
            nativeType='',
            type='',
        )

        col.nativeType = type(sa_col.type).__name__.upper()
        col.type = self.SA_TO_ATTR.get(col.nativeType, self.UNKNOWN_TYPE)

        return col
