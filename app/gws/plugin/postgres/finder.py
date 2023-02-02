import gws
import gws.base.database
import gws.base.model
import gws.base.search
import gws.types as t

from . import provider


@gws.ext.config.finder('postgres')
class Config(gws.base.search.finder.Config):
    """Database-based search"""

    db: t.Optional[str]
    """database provider uid"""
    tableName: str
    """sql table name"""


@gws.ext.object.finder('postgres')
class Object(gws.base.search.finder.Object):
    provider: provider.Object
    tableName: str

    def configure(self):
        self.tableName = self.var('tableName')
        self.configure_provider()
        self.configure_models()
        self.configure_templates()

        self.withKeyword = True
        self.withGeometry = True

        # if self.table.search_column:
        #     self.supportsKeyword = True
        # if self.table.geometry_column:
        #     self.supportsGeometry = True
        #
        # self.withKeyword = self.supportsKeyword and self.var('withKeyword', default=True)
        # self.withGeometry = self.supportsGeometry and self.var('withGeometry', default=True)

    def configure_provider(self):
        self.provider = gws.base.database.provider.get_for(self, ext_type='postgres')
        return True



    # def _filter_to_sql(self, f: gws.SearchFilter):
    #     if not f:
    #         return
    #
    #     if f.operator in ('and', 'or'):
    #         w = []
    #         p = []
    #         for sub in f.sub:
    #             where = self._filter_to_sql(sub)
    #             w.append(where[0])
    #             p.extend(where[1:])
    #         wstr = '(' + f' {f.operator} '.join(w) + ')'
    #         return [wstr, *p]
    #
    #     if f.operator == 'bbox':
    #         return [
    #             f'ST_Intersects(%s::geometry, "{self.table.geometry_column}")',
    #             f.shape.ewkb_hex
    #         ]
    #
    #     # @TODO must take editDataModel into account
    #
    #     return [f'{f.name} {f.operator} %s', f.value]
