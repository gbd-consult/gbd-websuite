import gws.common.search.provider
import gws.types as t


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
        self.db: t.SqlProviderObject = None
        self.keyword_required = False
        self.geometry_required = False

    def configure(self):
        super().configure()

        prov_uid = self.var('db')
        if prov_uid:
            self.db = self.root.find('gws.ext.db.provider', prov_uid)
        else:
            self.db = self.root.find_first('gws.ext.db.provider')

        self.keyword_required = self.var('keywordRequired')
        self.geometry_required = self.var('geometryRequired')

    def can_run(self, args):
        if self.keyword_required and not args.keyword:
            return False
        if self.geometry_required and not args.shapes:
            return False
        return args.keyword or args.shapes

    def run(self, layer: t.LayerObject, args: t.SearchArguments) -> t.List[t.Feature]:

        tab = self.var('table')

        fs = self.db.select(t.SelectArgs({
            'table': tab,
            'keyword': args.keyword,
            'shape': self.context_shape(args),
            'sort': self.var('sort'),
            'limit': args.limit,
            'tolerance': args.tolerance,
        }))

        sc = tab.get('searchColumn')

        for f in fs:
            f.category = self.var('title')
            f.title = f.attributes.get(sc)

        return fs
