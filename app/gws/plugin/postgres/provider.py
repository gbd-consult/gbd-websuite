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
    url: str

    def configure(self):
        self.url = self.configure_url()

    def configure_url(self):

        # https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
        # https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS

        defaults = {
            'application_name': 'GWS',
        }

        params = gws.merge(defaults, self.cfg('options'))

        p = self.cfg('host')
        if p:
            return gws.lib.net.make_url(
                scheme='postgresql',
                username=self.cfg('username'),
                password=self.cfg('password'),
                hostname=p,
                port=self.cfg('port'),
                path=self.cfg('database'),
                params=params,
            )

        p = self.cfg('serviceName')
        if p:
            s = os.getenv('PGSERVICEFILE')
            if not s or not os.path.isfile(s):
                raise gws.Error(f'PGSERVICEFILE not found')

            params['service'] = p

            return gws.lib.net.make_url(
                scheme='postgresql',
                hostname='',
                path=self.cfg('database', default=''),
                params=params,
            )

        raise gws.Error(f'"host/database" or "serviceName" are required')

    def engine(self, **kwargs):
        # kwargs.setdefault('poolclass', sa.NullPool)
        kwargs.setdefault('pool_pre_ping', True)
        return sa.create_engine(self.url, **kwargs)

    def qualified_table_name(self, table_name):
        if '.' in table_name:
            return table_name
        return 'public.' + table_name

    def split_table_name(self, table_name):
        if '.' in table_name:
            schema, name = table_name.split('.')
            return schema, name
        return 'public', table_name

    def join_table_name(self, table_name, schema=None):
        if not schema and '.' in table_name:
            schema, name = table_name.split('.')
        return schema + '.' + table_name

    def table_bounds(self, table_name):
        desc = self.describe(table_name)
        if not desc.geometryName:
            return
        tab = self.table(table_name)
        with self.session() as sess:
            sel = sa.select(sa.func.ST_Extent(tab.columns.get(desc.geometryName)))
            box = sess.execute(sel).scalar_one()
            if box:
                return gws.Bounds(extent=gws.gis.extent.from_box(box), crs=gws.gis.crs.get(desc.geometrySrid))
