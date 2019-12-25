import gws.common.search.provider
import gws.common.db

import gws.types as t

from . import provider, util


class Config(gws.common.search.provider.Config):
    """Database-based search"""

    db: t.Optional[str]  #: database provider uid
    table: t.SqlTableConfig  #: sql table configuration
    sort: t.Optional[str]  #: sort expression
    geometryRequired: bool = False
    keywordRequired: bool = False


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()
        self.provider: provider.Object = None
        self.table: t.SqlTable = None

    def configure(self):
        super().configure()

        self.provider: provider.Object = gws.common.db.require_provider(self, provider.Object)
        self.table = util.configure_table(self, self.provider)


    def can_run(self, args):
        if self.keyword_required and not args.keyword:
            return False
        if self.geometry_required and not args.shapes:
            return False
        return args.keyword or args.shapes

    def run(self, layer: t.LayerObject, args: t.SearchArguments) -> t.List[t.Feature]:

        return self.provider.select(t.SelectArgs({
            'table': self.table,
            'keyword': args.keyword,
            'shape': self.context_shape(args),
            'sort': self.var('sort'),
            'limit': args.limit,
            'tolerance': args.tolerance,
        }))
