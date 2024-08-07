from typing import Optional, cast

import gws
import gws.base.database
import gws.base.model
import gws.base.search
import gws.config.util

gws.ext.new.finder('postgres')


class Config(gws.base.search.finder.Config):
    """Database-based search"""

    dbUid: Optional[str]
    """database provider uid"""
    tableName: str
    """sql table name"""


class Object(gws.base.search.finder.Object):
    db: gws.DatabaseProvider
    tableName: str

    def configure(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        self.configure_provider()
        self.configure_models()
        self.configure_templates()

        # it's difficult to decide if we support keyword/geometry search,
        # because different models can have different rules

        self.supportsKeywordSearch = True
        self.supportsGeometrySearch = True
        self.supportsFilterSearch = True

    def configure_provider(self):
        return gws.config.util.configure_database_provider_for(self)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultDb=self.db,
            _defaultTableName=self.tableName
        )

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
