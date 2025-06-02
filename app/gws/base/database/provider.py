from typing import Optional, cast, Generator
import contextlib
import threading


import gws
import gws.lib.sa as sa

from . import connection


class Config(gws.Config):
    """Database provider"""

    schemaCacheLifeTime: gws.Duration = '3600'
    """Life time for schema caches."""
    withPool: Optional[bool] = False
    """Use connection pooling"""
    pool: Optional[dict]
    """Options for connection pooling."""


_thread_local = threading.local()


class Object(gws.DatabaseProvider):
    saEngine: sa.Engine
    saMetaMap: dict[str, sa.MetaData]

    def __getstate__(self):
        return gws.u.omit(vars(self), 'saMetaMap', 'saEngine')

    def configure(self):
        # init a dummy engine just to check things
        self.saEngine = self.create_engine(poolclass=sa.NullPool)
        self.saMetaMap = {}

    def activate(self):
        self.saEngine = self.create_engine()
        self.saMetaMap = {}

    def engine(self):
        eng = getattr(self, 'saEngine', None)
        if eng is not None:
            return eng
        self.saEngine = self.create_engine()
        return self.saEngine

    def create_engine(self, **kwargs):
        eng = sa.create_engine(self.url(), **self.engine_options(**kwargs))
        # setattr(eng, '_connection_cls', connection.Object)
        return eng

    def engine_options(self, **kwargs):
        if self.root.app.developer_option('db.engine_echo'):
            kwargs.setdefault('echo', True)
            kwargs.setdefault('echo_pool', True)

        if self.cfg('withPool') is False:
            kwargs.setdefault('poolclass', sa.NullPool)
            return kwargs

        pool = self.cfg('pool') or {}
        p = pool.get('disabled')
        if p is True:
            kwargs.setdefault('poolclass', sa.NullPool)
            return kwargs

        p = pool.get('pre_ping')
        if p is True:
            kwargs.setdefault('pool_pre_ping', True)
        p = pool.get('size')
        if isinstance(p, int):
            kwargs.setdefault('pool_size', p)
        p = pool.get('recycle')
        if isinstance(p, int):
            kwargs.setdefault('pool_recycle', p)
        p = pool.get('timeout')
        if isinstance(p, int):
            kwargs.setdefault('pool_timeout', p)

        return kwargs

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
                md.reflect(conn.saConn, schema, resolve_fks=False, views=True)
            gws.debug.time_end()
            return md

        life_time = self.cfg('schemaCacheLifeTime', 0)
        if not life_time:
            self.saMetaMap[schema] = _load()
        else:
            self.saMetaMap[schema] = gws.u.get_cached_object(f'database_metadata_schema_{schema}', life_time, _load)

    def connect(self):
        conn = self._open_connection()
        return connection.Object(self, conn)

    def _sa_connection(self) -> sa.Connection | None:
        return getattr(_thread_local, '_connection', None)

    def _open_connection(self) -> sa.Connection:
        conn = getattr(_thread_local, '_connection', None)
        cc = getattr(_thread_local, '_connectionCount', 0)

        if conn is None:
            assert cc == 0
            conn = self.engine().connect()
            setattr(_thread_local, '_connection', conn)
        else:
            assert cc > 0
        
        setattr(_thread_local, '_connectionCount', cc + 1)
        gws.log.debug(f'db.connect: open: {cc + 1}')
        return conn

    def _close_connection(self):
        conn = getattr(_thread_local, '_connection', None)
        cc = getattr(_thread_local, '_connectionCount', 0)
        assert conn is not None
        assert cc > 0
        gws.log.debug(f'db.connect: close: {cc}')
        if cc == 1:
            if conn:
                conn.close()
            setattr(_thread_local, '_connection', None)
            setattr(_thread_local, '_connectionCount', 0)
        else:
            setattr(_thread_local, '_connectionCount', cc - 1)

    def table(self, table, **kwargs):
        tab = self._sa_table(table)
        if tab is None:
            raise sa.Error(f'table {str(table)} not found')
        return tab

    def count(self, table):
        tab = self._sa_table(table)
        if tab is None:
            return 0
        sql = sa.select(sa.func.count()).select_from(tab)
        with self.connect() as conn:
            return conn.fetch_int(sql)

    def has_table(self, table_name: str):
        tab = self._sa_table(table_name)
        return tab is not None

    def _sa_table(self, tab_or_name) -> sa.Table | None:
        if isinstance(tab_or_name, sa.Table):
            return tab_or_name
        schema, name = self.split_table_name(tab_or_name)
        self.reflect_schema(schema)
        # see _get_table_key in sqlalchemy/sql/schema.py
        table_key = schema + '.' + name
        sm = self.saMetaMap[schema]
        if sm is None:
            raise sa.Error(f'schema {schema!r} not found')
        return sm.tables.get(table_key)

    def column(self, table, column_name):
        tab = self.table(table)
        try:
            return tab.columns[column_name]
        except KeyError:
            raise sa.Error(f'column {str(table)}.{column_name!r} not found')

    def has_column(self, table, column_name):
        tab = self._sa_table(table)
        return tab is not None and column_name in tab.columns

    def select_text(self, sql, **kwargs):
        with self.connect() as conn:
            try:
                return [gws.u.to_dict(r) for r in conn.execute(sa.text(sql), kwargs)]
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
        tab = self._sa_table(table)
        if tab is None:
            raise sa.Error(f'table {table!r} not found')

        schema = tab.schema
        name = tab.name

        desc = gws.DataSetDescription(
            columns=[],
            columnMap={},
            fullName=self.join_table_name(schema or '', name),
            geometryName='',
            geometrySrid=0,
            geometryType='',
            name=name,
            schema=schema,
        )

        for n, sa_col in enumerate(cast(list[sa.Column], tab.columns)):
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


##
