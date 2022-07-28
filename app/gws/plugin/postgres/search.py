import gws
import gws.base.db
import gws.base.search
import gws.types as t

from . import provider


@gws.ext.config.finder('postgres')
class Config(gws.base.search.provider.Config):
    """Database-based search"""

    db: t.Optional[str]  #: database provider uid
    table: gws.base.db.SqlTableConfig  #: sql table configuration
    sort: t.Optional[str]  #: sort expression


@gws.ext.object.finder('postgres')
class Object(gws.base.search.provider.Object):
    provider: provider.Object
    table: gws.SqlTable

    def configure(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.table = self.var('_table')
        else:
            self.provider = provider.require_for(self)
            self.table = self.provider.configure_table(self.var('table'))

        if self.table.search_column:
            self.supports_keyword = True
        if self.table.geometry_column:
            self.supports_geometry = True

    def run(self, args, layer=None):
        n, u = args.tolerance or self.tolerance
        geometry_tolerance = n * args.resolution if u == 'px' else n
        return self.provider.select_features(gws.SqlSelectArgs(
            table=self.table,
            keyword=args.keyword,
            shape=self.context_shape(args),
            sort=self.var('sort'),
            limit=args.limit,
            geometry_tolerance=geometry_tolerance,
            extra_where=self._filter_to_sql(args.filter),
        ))

    def _filter_to_sql(self, f: gws.SearchFilter):
        if not f:
            return

        if f.operator in ('and', 'or'):
            w = []
            p = []
            for sub in f.sub:
                where = self._filter_to_sql(sub)
                w.append(where[0])
                p.extend(where[1:])
            wstr = '(' + f' {f.operator} '.join(w) + ')'
            return [wstr, *p]

        if f.operator == 'bbox':
            return [
                f'ST_Intersects(%s::geometry, "{self.table.geometry_column}")',
                f.shape.ewkb_hex
            ]

        # @TODO must take editDataModel into account

        return [f'{f.name} {f.operator} %s', f.value]
