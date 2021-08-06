"""Direct QGIS/Postgres search."""

import gws
import gws.config.parser
import gws.common.search.provider
import gws.ext.db.provider.postgres
import gws.types as t
import gws.core.tree


class Config(gws.common.search.provider.Config):
    dataSource: dict  #: qgis data source


class Object(gws.common.search.provider.Object):
    def configure(self):
        super().configure()

        self.capabilties = gws.common.search.provider.CAPS_GEOMETRY

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
            self.table: t.SqlTable = self.db.configure_table(t.Config(name=ds['table'], geometryColumn=ds.get('geometryColumn')))
        except gws.Error:
            gws.log.warn(f"table {ds['table']!r} not found or invalid")
            gws.log.exception()
            self.active = False

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        if not self.table:
            return []
        n, u = args.tolerance or self.tolerance
        map_tolerance = n * args.resolution if u == 'px' else n
        return self.db.select(t.SelectArgs(
            table=self.table,
            shape=self.context_shape(args),
            limit=args.limit,
            map_tolerance=map_tolerance,
            extra_where=self.extra_where,
        ))
