"""Postgres database provider."""

import os

import gws.base.database
import gws.gis.crs
import gws.gis.extent
import gws.lib.net
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.databaseProvider('postgres')


class Config(gws.base.database.provider.Config):
    """Postgres/Postgis database provider"""

    database: t.Optional[str]
    """database name"""
    host: t.Optional[str]
    """database host"""
    port: int = 5432
    """database port"""
    username: t.Optional[str]
    """username"""
    password: t.Optional[str]
    """password"""
    serviceName: t.Optional[str]
    """service name from pg_services file"""
    options: t.Optional[dict]
    """connection options"""


class Object(gws.base.database.provider.Object):
    def configure(self):
        self.url = connection_url(self.config)
        if not self.url:
            raise gws.Error(f'"host/database" or "serviceName" are required')

    def engine(self, **kwargs):
        # kwargs.setdefault('poolclass', sa.NullPool)
        # kwargs.setdefault('pool_pre_ping', True)
        kwargs.setdefault('echo', self.root.app.developer_option('db.engine_echo'))
        url = connection_url(self.config)
        return sa.create_engine(url, **kwargs)

    def qualified_table_name(self, table_name):
        if '.' in table_name:
            return table_name
        return 'public.' + table_name

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

    def table_bounds(self, table_name):
        desc = self.describe(table_name)
        if not desc.geometryName:
            return
        tab = self.table(table_name)

        sel = sa.select(sa.func.ST_Extent(tab.columns.get(desc.geometryName)))
        with self.connection() as conn:
            box = conn.execute(sel).scalar_one()
            if box:
                return gws.Bounds(extent=gws.gis.extent.from_box(box), crs=gws.gis.crs.get(desc.geometrySrid))


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
            raise gws.Error(f'PGSERVICEFILE {s!r} not found')

        params['service'] = p

        return gws.lib.net.make_url(
            scheme='postgresql',
            hostname='',
            path=cfg.get('database') or cfg.get('dbname') or '',
            params=params,
        )
