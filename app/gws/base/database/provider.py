import gws.lib.sa as sa

import gws
import gws.types as t

from . import manager


class Config(gws.Config):
    """Database provider"""
    pass


class Object(gws.Node, gws.IDatabaseProvider):
    mgr: manager.Object
    saEngine: sa.Engine
    saMetaMap: dict[str, sa.MetaData]

    def __getstate__(self):
        return gws.omit(vars(self), 'saMetaMap', 'saEngine')

    def configure(self):
        self.mgr = self.cfg('_defaultManager')
        self.models = []

        self.saEngine = self.engine()
        self.saMetaMap = {}

    def activate(self):
        self.saEngine = self.engine()
        self.saMetaMap = {}

    def autoload(self, schema: str):
        if schema in self.saMetaMap:
            return

        self.saMetaMap[schema] = sa.MetaData(schema=schema)

        # introspecting the whole schema is generally faster
        # but what if we only need a single table from a big schema?
        # @TODO add options for reflection

        gws.time_start(f'AUTOLOAD {self.uid=} {schema=}')
        self.saMetaMap[schema].reflect(self.saEngine, schema, resolve_fks=False, views=True)
        gws.time_end()

    def connection(self) -> sa.Connection:
        return self.saEngine.connect()

    def table(self, table_name: str, **kwargs):
        tab = self._table(table_name)
        if tab is None:
            raise gws.Error(f'table {table_name!r} not found')
        return tab

    def has_table(self, table_name: str):
        tab = self._table(table_name)
        return tab is not None

    def _table(self, table_name) -> sa.Table:
        schema, name = self.split_table_name(table_name)
        self.autoload(schema)
        return self.saMetaMap[schema].tables.get(self.join_table_name(schema, name))

    def column(self, table, column_name):
        try:
            return table.columns[column_name]
        except KeyError:
            raise gws.Error(f'column {table.name!r}.{column_name!r} not found')

    def has_column(self, table, column_name):
        return column_name in table.columns

    # https://www.psycopg.org/docs/usage.html#adaptation-of-python-values-to-sql-types

    SA_TO_ATTR = {
        'ARRAY': gws.AttributeType.strlist,
        'BIGINT': gws.AttributeType.int,
        'BIGSERIAL': gws.AttributeType.int,
        'BIT': gws.AttributeType.int,
        'BOOL': gws.AttributeType.bool,
        'BOOLEAN': gws.AttributeType.bool,
        'BYTEA': gws.AttributeType.bytes,
        'CHAR': gws.AttributeType.str,
        'CHARACTER VARYING': gws.AttributeType.str,
        'CHARACTER': gws.AttributeType.str,
        'DATE': gws.AttributeType.date,
        'DECIMAL': gws.AttributeType.float,
        'DOUBLE PRECISION': gws.AttributeType.float,
        'FLOAT4': gws.AttributeType.float,
        'FLOAT8': gws.AttributeType.float,
        'GEOMETRY': gws.AttributeType.geometry,
        'INT': gws.AttributeType.int,
        'INT2': gws.AttributeType.int,
        'INT4': gws.AttributeType.int,
        'INT8': gws.AttributeType.int,
        'INTEGER': gws.AttributeType.int,
        'MONEY': gws.AttributeType.float,
        'NUMERIC': gws.AttributeType.float,
        'REAL': gws.AttributeType.float,
        'SERIAL': gws.AttributeType.int,
        'SERIAL2': gws.AttributeType.int,
        'SERIAL4': gws.AttributeType.int,
        'SERIAL8': gws.AttributeType.int,
        'SMALLINT': gws.AttributeType.int,
        'SMALLSERIAL': gws.AttributeType.int,
        'TEXT': gws.AttributeType.str,
        'TIME': gws.AttributeType.time,
        'TIMESTAMP': gws.AttributeType.datetime,
        'TIMESTAMPTZ': gws.AttributeType.datetime,
        'TIMETZ': gws.AttributeType.time,
        'VARCHAR': gws.AttributeType.str,
    }
    SA_TO_GEOM = {
        'GEOMETRY': gws.GeometryType.geometry,
        'POINT': gws.GeometryType.point,
        'LINESTRING': gws.GeometryType.linestring,
        'POLYGON': gws.GeometryType.polygon,
        'MULTIPOINT': gws.GeometryType.multipoint,
        'MULTILINESTRING': gws.GeometryType.multilinestring,
        'MULTIPOLYGON': gws.GeometryType.multipolygon,
        'GEOMETRYCOLLECTION': gws.GeometryType.geometrycollection,
        'CURVE': gws.GeometryType.curve,
    }

    def describe(self, table_name: str):
        table = self._table(table_name)
        if table is None:
            return

        schema, name = self.split_table_name(table_name)

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

        for n, c in enumerate(t.cast(list[sa.Column], table.columns)):
            col = gws.ColumnDescription(
                columnIndex=n,
                comment=str(c.comment or ''),
                default=c.default,
                geometrySrid=0,
                geometryType='',
                isAutoincrement=bool(c.autoincrement),
                isNullable=bool(c.nullable),
                isPrimaryKey=bool(c.primary_key),
                isUnique=bool(c.unique),
                hasDefault=c.server_default is not None,
                name=str(c.name),
                nativeType='',
                type='',
            )

            typ = c.type
            col.nativeType = str(typ).upper()

            gt = getattr(typ, 'geometry_type', None)
            if gt:
                col.type = gws.AttributeType.geometry
                col.geometryType = gt.lower()
                col.geometrySrid = getattr(typ, 'srid')
            else:
                col.type = self.SA_TO_ATTR.get(col.nativeType, gws.AttributeType.str)

            desc.columns.append(col)
            desc.columnMap[col.name] = col

        for col in desc.columns:
            if col.geometryType:
                desc.geometryName = col.name
                desc.geometryType = col.geometryType
                desc.geometrySrid = col.geometrySrid
                break

        return desc


def get_for(obj: gws.INode, uid: str = None, ext_type: str = None):
    mgr = obj.root.app.databaseMgr

    uid = uid or obj.cfg('dbUid')
    if uid:
        p = mgr.provider(uid)
        if not p:
            raise gws.Error(f'database provider {uid!r} not found')
        return p

    if obj.cfg('_defaultProvider'):
        return obj.cfg('_defaultProvider')

    ext_type = ext_type or obj.extType
    p = mgr.first_provider(ext_type)
    if not p:
        raise gws.Error(f'no database providers of type {ext_type!r} found')
    return p
