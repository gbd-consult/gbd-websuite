"""Base database provider."""

import contextlib
import threading
from typing import Optional, cast

import gws
import gws.lib.sa as sa

from . import connection


class Config(gws.Config):
    """Database provider"""

    schemaCacheLifeTime: gws.Duration = '3600'
    """How long table structures read from the database are cached."""
    withPool: Optional[bool] = False
    """Keep and reuse database connections in a pool."""
    pool: Optional[dict]
    """Connection pool options."""


_thread_local = threading.local()


def _connections() -> dict[str, sa.Connection]:
    """Return the open connections of the current thread, keyed by provider uid."""
    d = getattr(_thread_local, 'connections', None)
    if d is None:
        d = {}
        _thread_local.connections = d
    return d


class Object(gws.DatabaseProvider):
    """Base database provider.

    Manages the SQLAlchemy engine and the per-thread connection, reflects table
    structures and describes tables and columns, and runs plain SQL text.

    Subclasses provide ``url``, ``split_table_name``, ``join_table_name`` and
    ``table_bounds``, and extend ``describe_column`` for database-specific types.
    """

    saEngine: sa.Engine
    """SQLAlchemy engine."""
    saMetaMap: dict[str, sa.MetaData]
    """Reflected metadata, keyed by schema name."""

    def __getstate__(self):
        """Return the state for pickling, without the engine and the metadata."""
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

    def inspect_schema(self, schema, options=None):
        if options and options.refresh:
            self.saMetaMap.pop(schema, None)

        if schema in self.saMetaMap:
            return

        def _load():
            md = sa.MetaData(schema=schema)

            # introspecting the whole schema is generally faster
            # but what if we only need a single table from a big schema?
            # @TODO add options for reflection

            gws.debug.time_start(f'AUTOLOAD {self.uid=} {schema=}')
            with self.begin() as conn:
                md.reflect(conn.saConn, schema, resolve_fks=False, views=True)
            gws.debug.time_end()
            return md

        life_time = self.cfg('schemaCacheLifeTime', 0)
        if options and options.cacheLifeTime is not None:
            life_time = options.cacheLifeTime
        if not life_time:
            self.saMetaMap[schema] = _load()
        else:
            self.saMetaMap[schema] = gws.u.get_cached_object(f'database_metadata_schema_{schema}', life_time, _load)

    @contextlib.contextmanager
    def begin(self, nested=False):
        conns = _connections()
        sa_conn = conns.get(self.uid)

        if sa_conn is not None:
            if not nested:
                yield connection.Object(self, sa_conn)
                return
            with sa_conn.begin_nested():
                yield connection.Object(self, sa_conn)
            return

        sa_conn = self.engine().connect()
        conns[self.uid] = sa_conn
        try:
            with sa_conn.begin():
                yield connection.Object(self, sa_conn)
        finally:
            conns.pop(self.uid, None)
            sa_conn.close()

    def connect(self):
        return self.begin()

    @contextlib.contextmanager
    def autocommit_connection(self):
        if self.uid in _connections():
            raise gws.Error(f'db {self.uid!r}: autocommit_connection inside a transaction')
        sa_conn = self.engine().connect().execution_options(isolation_level='AUTOCOMMIT')
        try:
            yield connection.Object(self, sa_conn)
        finally:
            sa_conn.close()

    def _sa_connection(self) -> sa.Connection | None:
        """Return the open connection of the current thread, if any."""
        return _connections().get(self.uid)

    def table(self, table, **kwargs):
        tab = self._sa_table(table)
        if tab is None:
            raise sa.Error(f'table not found: {table!r}')
        return tab

    def count(self, table):
        tab = self._sa_table(table)
        if tab is None:
            return 0
        sql = sa.select(sa.func.count()).select_from(tab)
        with self.begin() as conn:
            return conn.fetch_int(sql)

    def has_schema(self, schema):
        return schema in self.schema_names()

    def schema_names(self):
        inspector = sa.inspect(self.engine())
        return inspector.get_schema_names()
    
    def has_table(self, table_name: str):
        tab = self._sa_table(table_name)
        return tab is not None

    def _sa_table(self, tab_or_name) -> sa.Table | None:
        """Return a reflected table by name, or ``None`` if the table does not exist."""
        if isinstance(tab_or_name, sa.Table):
            return tab_or_name
        schema, name = self.split_table_name(tab_or_name)
        self.inspect_schema(schema)
        # see _get_table_key in sqlalchemy/sql/schema.py
        table_key = schema + '.' + name
        sm = self.saMetaMap.get(schema)
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
        with self.begin() as conn:
            return [gws.u.to_dict(r) for r in conn.execute(sa.text(sql), kwargs)]

    def execute_text(self, sql, **kwargs):
        with self.begin() as conn:
            return conn.execute(sa.text(sql), kwargs)

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
    """Attribute types for SQLAlchemy type names."""

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
    """Geometry types for database geometry type names."""

    UNKNOWN_TYPE = gws.AttributeType.str
    """Attribute type for columns of unknown types."""
    UNKNOWN_ARRAY_TYPE = gws.AttributeType.strlist
    """Attribute type for array columns of unknown item types."""

    def describe(self, table):
        tab = self._sa_table(table)
        if tab is None:
            raise sa.Error(f'table not found: {table!r}')

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

    def describe_column(self, table, column_name):
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
