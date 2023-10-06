import gws.common.search.provider
import gws.common.db

import gws.types as t

from . import provider


class Config(gws.common.search.provider.Config):
    """Database-based search"""

    db: t.Optional[str]  #: database provider uid
    table: gws.common.db.SqlTableConfig  #: sql table configuration
    sort: t.Optional[str]  #: sort expression
    keywordColumns: t.Optional[t.List[str]]


class Object(gws.common.search.provider.Object):
    def configure(self):
        super().configure()

        self.provider: provider.Object = t.cast(provider.Object, gws.common.db.require_provider(self, provider.Object))
        try:
            self.table: t.SqlTable = self.provider.configure_table(self.var('table'))
        except gws.Error:
            gws.log.exception()
            self.active = False
            return

        self.capabilties = gws.common.search.provider.CAPS_FILTER
        if self.table.geometry_column:
            self.capabilties |= gws.common.search.provider.CAPS_GEOMETRY

        self.keyword_columns = self.var('keywordColumns')
        if self.keyword_columns:
            self.capabilties |= gws.common.search.provider.CAPS_KEYWORD

    def run(self, req: t.IRequest, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        n, u = args.tolerance or self.tolerance
        map_tolerance = n * args.resolution if u == 'px' else n

        model = self.model or layer.model
        args = t.SelectArgs(
            table=self.table,
            keyword=args.keyword,
            keyword_columns=self.keyword_columns,
            shape=self.context_shape(args),
            sort=self.var('sort'),
            limit=args.limit,
            map_tolerance=map_tolerance,
            extra_where=self._filter_to_sql(args.filter),
            depth=args.relation_depth or 0,
        )
        return model.select(args, t.ModelContext(user=args.user, depth=args.depth))

    def _filter_to_sql(self, f: t.SearchFilter):
        if not f:
            return

        if f.operator in ('and', 'or'):
            w = []
            p = []
            for sub in f.sub:
                where = self._filter_to_sql(sub)
                w.append(where[0])
                p.extend(where[1:])
            w = '(' + f' {f.operator} '.join(w) + ')'
            return [w, *p]

        if f.op == 'bbox':
            return [
                f'ST_Intersects(%s::geometry, "{self.table.geometry_column}")',
                f.shape.ewkb_hex
            ]

        # @TODO must take editDataModel into account

        return [f'{f.name} {f.operator} %s', f.value]
