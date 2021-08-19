"""Direct QGIS/Postgres search."""

import gws.ext.db.provider.postgres

import gws
import gws.base.search
import gws.core.tree
import gws.types as t


class Config(gws.base.search.provider.Config):
    dataSource: dict  #: qgis data source


class Object(gws.base.search.provider.Object):
    def configure(self):
        

        self.capabilties = gws.base.search.provider.CAPS_GEOMETRY

        ds = self.var('dataSource')

        cfg = {
            'database': ds['dbname'],
            'host': ds['host'],
            'port': ds['port'],
            'user': ds['user'],
            'password': ds['password'],
        }

        self.extra_where = None
        sql = ds.get('sql')
        if sql:
            self.extra_where = [sql.replace('%', '%%')]

        db_uid = gws.as_uid(f"h={cfg['host']}_p={cfg['port']}_u={cfg['user']}_d={cfg['database']}")

        self.db = t.cast(
            gws.ext.db.provider.postgres.Object,
            self.root.create_shared_object('gws.ext.db.provider.postgres', db_uid, cfg))

        try:
            self.table: gws.SqlTable = self.db.configure_table(gws.Config(name=ds['table'], geometryColumn=ds['geometryColumn']))
        except gws.Error:
            gws.log.exception(f"table {ds['table']!r} not found or invalid")
            self.active = False

    def run(self, args, layer=None):
        if not self.table:
            return []
        n, u = args.tolerance or self.tolerance
        map_tolerance = n * args.resolution if u == 'px' else n
        return self.db.select(gws.SqlSelectArgs(
            table=self.table,
            shape=self.context_shape(args),
            limit=args.limit,
            map_tolerance=map_tolerance,
            extra_where=self.extra_where,
        ))
