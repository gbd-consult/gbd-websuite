from typing import Optional, cast

import gws.lib.sa as sa

import gws

from . import manager


class Config(gws.Config):
    """Database provider"""

    schemaCacheLifeTime: gws.Duration = 3600
    """life time for schema caches"""


class Object(gws.DatabaseProvider):
    saEngine: sa.Engine
    saMetaMap: dict[str, sa.MetaData]

    def __getstate__(self):
        return gws.u.omit(vars(self), 'saMetaMap', 'saEngine')

    def configure(self):
        self.models = []
        self.saEngine = self.engine()
        self.saMetaMap = {}

    def activate(self):
        self.saEngine = self.engine()
        self.saMetaMap = {}

    def autoload(self, schema: str):
        if schema in self.saMetaMap:
            return

        def _load():
            md = sa.MetaData(schema=schema)

            # introspecting the whole schema is generally faster
            # but what if we only need a single table from a big schema?
            # @TODO add options for reflection

            gws.debug.time_start(f'AUTOLOAD {self.uid=} {schema=}')
            with self.connection() as conn:
                md.reflect(conn, schema, resolve_fks=False, views=True)
            gws.debug.time_end()

            return md

        life_time = self.cfg('schemaCacheLifeTime', 0)
        if not life_time:
            self.saMetaMap[schema] = _load()
        else:
            self.saMetaMap[schema] = gws.u.get_cached_object(f'database_metadata_schema_{schema}', life_time, _load)

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

        for n, c in enumerate(cast(list[sa.Column], table.columns)):
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

            col.nativeType = type(c.type).__name__.upper()

            if col.nativeType == 'ARRAY':
                it = getattr(c.type, 'item_type', None)
                ia = self.SA_TO_ATTR.get(type(it).__name__.upper())
                if ia == gws.AttributeType.str:
                    col.type = gws.AttributeType.strlist
                elif ia == gws.AttributeType.int:
                    col.type = gws.AttributeType.intlist
                elif ia == gws.AttributeType.float:
                    col.type = gws.AttributeType.floatlist
                else:
                    col.type = self.UNKNOWN_ARRAY_TYPE

            elif col.nativeType == 'GEOMETRY':
                gt = getattr(c.type, 'geometry_type', '').upper()
                ga = self.SA_TO_GEOM.get(gt, gws.GeometryType.geometry)
                col.type = gws.AttributeType.geometry
                col.geometryType = ga
                col.geometrySrid = getattr(c.type, 'srid', 0)

            else:
                col.type = self.SA_TO_ATTR.get(col.nativeType, self.UNKNOWN_TYPE)

            desc.columns.append(col)
            desc.columnMap[col.name] = col

        for col in desc.columns:
            if col.geometryType:
                desc.geometryName = col.name
                desc.geometryType = col.geometryType
                desc.geometrySrid = col.geometrySrid
                break

        return desc
