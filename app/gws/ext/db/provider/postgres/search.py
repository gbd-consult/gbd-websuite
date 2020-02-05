import gws.common.search.provider
import gws.common.db

import gws.types as t

from . import provider


class Config(gws.common.search.provider.Config):
    """Database-based search"""

    db: t.Optional[str]  #: database provider uid
    table: gws.common.db.SqlTableConfig  #: sql table configuration
    sort: t.Optional[str]  #: sort expression


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()
        self.provider: provider.Object = None
        self.table: t.SqlTable = None

    def configure(self):
        super().configure()

        self.provider: provider.Object = gws.common.db.require_provider(self, provider.Object)
        self.table = self.provider.configure_table(self.var('table'))

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        n, u = args.tolerance or self.tolerance
        map_tolerance = n * args.resolution if u == 'px' else n
        return self.provider.select(t.SelectArgs(
            table=self.table,
            keyword=args.keyword,
            shape=self.context_shape(args),
            sort=self.var('sort'),
            limit=args.limit,
            map_tolerance=map_tolerance,
        ))
