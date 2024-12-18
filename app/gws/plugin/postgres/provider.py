"""Postgres database provider."""

from typing import Optional

import os
import re

import gws.base.database
import gws.lib.crs
import gws.lib.extent
import gws.lib.net
import gws.lib.sa as sa

gws.ext.new.databaseProvider('postgres')


class Config(gws.base.database.provider.Config):
    """Postgres/Postgis database provider"""

    database: Optional[str]
    """Database name."""
    host: Optional[str]
    """Database host."""
    port: int = 5432
    """Database port."""
    username: Optional[str]
    """Username."""
    password: Optional[str]
    """Password."""
    serviceName: Optional[str]
    """Service name from pg_services file."""
    options: Optional[dict]
    """Libpq connection options."""
    pool: Optional[dict]
    """Options for connection pooling."""


class Object(gws.base.database.provider.Object):
    def configure(self):
        self.url = connection_url(self.config)
        if not self.url:
            raise sa.Error(f'"host/database" or "serviceName" are required')

    def engine(self, **kwargs):
        pool = self.cfg('pool') or {}
        p = pool.get('disabled')
        if p is True:
            kwargs.setdefault('poolclass', sa.NullPool)
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

        if self.root.app.developer_option('db.engine_echo'):
            kwargs.setdefault('echo', True)
            kwargs.setdefault('echo_pool', True)

        url = connection_url(self.config)
        return sa.create_engine(url, **kwargs)

    _RE_TABLE_NAME = r'''(?x) 
        ^
        (
            ( " (?P<a1> ([^"] | "")+ ) " )
            |
            (?P<a2> [^".]+ )
        )
        (
            \.
            (
                ( " (?P<b1> ([^"] | "")+ ) " )
                |
                (?P<b2> [^".]+ )
            )
        )?
        $
    '''

    _DEFAULT_SCHEMA = 'public'

    def split_table_name(self, table_name):
        m = re.match(self._RE_TABLE_NAME, table_name.strip())
        if not m:
            raise ValueError(f'invalid table name {table_name!r}')

        d = m.groupdict()
        s = d['a1'] or d['a2']
        t = d['b1'] or d['b2']
        if not t:
            s, t = self._DEFAULT_SCHEMA, s

        return s.replace('""', '"'), t.replace('""', '"')

    def join_table_name(self, schema, name):
        if schema:
            return schema + '.' + name
        schema, name2 = self.split_table_name(name)
        return schema + '.' + name2

    def table_bounds(self, table):
        desc = self.describe(table)
        if not desc.geometryName:
            return

        tab = self.table(table)
        sql = sa.select(sa.func.ST_Extent(tab.columns.get(desc.geometryName)))
        with self.connect() as conn:
            box = conn.execute(sql).scalar_one()
        extent = gws.lib.extent.from_box(box)
        if extent:
            return gws.Bounds(extent=extent, crs=gws.lib.crs.get(desc.geometrySrid))

    def describe_column(self, table, column_name):
        col = super().describe_column(table, column_name)

        if col.nativeType == 'ARRAY':
            sa_col = self.column(table, column_name)
            it = getattr(sa_col.type, 'item_type', None)
            ia = self.SA_TO_ATTR.get(type(it).__name__.upper())
            if ia == gws.AttributeType.str:
                col.type = gws.AttributeType.strlist
            elif ia == gws.AttributeType.int:
                col.type = gws.AttributeType.intlist
            elif ia == gws.AttributeType.float:
                col.type = gws.AttributeType.floatlist
            else:
                col.type = self.UNKNOWN_ARRAY_TYPE
            return col

        if col.nativeType == 'GEOMETRY':
            typ, srid = self._get_geom_type_and_srid(table, column_name)
            col.type = gws.AttributeType.geometry
            col.geometryType = self.SA_TO_GEOM.get(typ, gws.GeometryType.geometry)
            col.geometrySrid = srid
            return col

        return col

    def _get_geom_type_and_srid(self, table, column_name):
        sa_table = self.table(table)
        sa_col = self.column(table, column_name)

        typ = getattr(sa_col.type, 'geometry_type', '').upper()
        srid = getattr(sa_col.type, 'srid', 0)

        if typ != 'GEOMETRY' and srid > 0:
            return typ, srid

        # not a typmod, possibly constraint-based. Query "geometry_columns"...

        gcs = getattr(self, '_geometry_columns_cache', None)
        if not gcs:
            gcs = self.select_text(f'''
                SELECT  
                    f_table_schema,
                    f_table_name,
                    f_geometry_column,
                    type,
                    srid
                FROM public.geometry_columns
            ''')
            setattr(self, '_geometry_columns_cache', gcs)

        for gc in gcs:
            if (
                    gc['f_table_schema'] == sa_table.schema
                    and gc['f_table_name'] == sa_table.name
                    and gc['f_geometry_column'] == sa_col.name
            ):
                return gc['type'], gc['srid']

        return 'GEOMETRY', -1


##

def connection_url(cfg: gws.Config):
    # https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
    # https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS

    defaults = {
        'application_name': 'GWS',
    }

    params = gws.u.merge(defaults, cfg.get('options'))

    p = cfg.get('host')
    if p:
        return gws.lib.net.make_url(
            scheme='postgresql',
            username=cfg.get('username'),
            password=cfg.get('password'),
            hostname=p,
            port=cfg.get('port'),
            path=cfg.get('database') or cfg.get('dbname') or '',
            params=params,
        )

    p = cfg.get('serviceName')
    if p:
        s = os.getenv('PGSERVICEFILE')
        if not s or not os.path.isfile(s):
            raise sa.Error(f'PGSERVICEFILE {s!r} not found')

        params['service'] = p

        return gws.lib.net.make_url(
            scheme='postgresql',
            hostname='',
            path=cfg.get('database') or cfg.get('dbname') or '',
            params=params,
        )
