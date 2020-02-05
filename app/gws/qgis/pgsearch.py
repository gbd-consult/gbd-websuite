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
    def __init__(self):
        super().__init__()
        self.db: gws.ext.db.provider.postgres.Object = None
        self.table: t.SqlTable = None
        self.extra_where = ''

    def configure(self):
        super().configure()

        ds = self.var('ds')

        cfg = {
            'database': ds['dbname'],
            'host': ds['host'],
            'port': ds['port'],
            'user': ds['user'],
            'password': ds['password'],
        }

        self.extra_where = ds.get('sql')

        self.db = self.create_shared_object(
            'gws.ext.db.provider.postgres',
            gws.as_uid(f"h={cfg['host']}_p={cfg['port']}_u={cfg['user']}_d={cfg['database']}"),
            t.Config(cfg))

        self.table = t.SqlTable({
            'geometryColumn': ds['geometryColumn'],
            'name': ds['table'],
        })

    def can_run(self, args):
        # qgis-pg is for spatial searches only
        return args.shapes and not args.keyword

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        n, u = args.tolerance or self.tolerance
        map_tolerance = n * args.resolution if u == 'px' else n
        return self.db.select(t.SelectArgs(
            table=self.table,
            shape=self.context_shape(args),
            limit=args.limit,
            map_tolerance=map_tolerance,
            extra_where=self.extra_where,
        ))
