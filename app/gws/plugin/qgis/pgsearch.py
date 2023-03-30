"""Direct QGIS/Postgres search."""

import gws
import gws.base.database.postgres.provider
import gws.base.search
import gws.types as t


gws.ext.new.finder('qgispostgres')

class Object(gws.base.search.provider.Object):
    supports_geometry = True
    supports_keyword = True

    extra_where: list[str]
    db: gws.base.db.postgres.provider.Object
    table: gws.SqlTable

    def configure(self):
        ds = self.var('_data_source')
        cfg = {
            'database': ds['dbname'],
            'host': ds['host'],
            'port': ds['port'],
            'user': ds['user'],
            'password': ds['password'],
        }

        self.extra_where = []
        sql = ds.get('sql')
        if sql:
            self.extra_where = [sql.replace('%', '%%')]

        self.db = gws.base.db.postgres.provider.create(self.root, cfg, shared=True)
        self.table = self.db.configure_table(gws.Config(name=ds.get('table'), geometryColumn=ds.get('geometryColumn')))

    def run(self, args, layer=None):
        n, u = args.tolerance or self.tolerance
        geometry_tolerance = n * args.resolution if u == 'px' else n
        return self.db.select(gws.SqlSelectArgs(
            table=self.table,
            keyword=args.keyword,
            shape=self.context_shape(args),
            limit=args.limit,
            geometry_tolerance=geometry_tolerance,
            extra_where=self.extra_where,
        ))
