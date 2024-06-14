"""Postgres database provider."""

from typing import Optional

import os

import gws.base.database
import gws.gis.crs
import gws.gis.extent
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

    def split_table_name(self, table_name):
        if '.' in table_name:
            schema, _, name = table_name.partition('.')
            return schema, name
        return 'public', table_name

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
        extent = gws.gis.extent.from_box(box)
        if extent:
            return gws.Bounds(extent=extent, crs=gws.gis.crs.get(desc.geometrySrid))


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
