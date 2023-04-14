import gws
import gws.base.database
import gws.base.layer.vector
import gws.gis.bounds
import gws.base.feature
import gws.gis.crs
import gws.base.shape
import gws.types as t

from . import provider

gws.ext.new.layer('postgres')


class Config(gws.base.layer.Config):
    """Postgres layer"""

    db: t.Optional[str]
    """database provider uid"""
    tableName: str
    """sql table name"""


class Object(gws.base.layer.vector.Object):
    provider: provider.Object
    tableName: str
    tableDesc: gws.DataSetDescription

    def configure(self):
        self.configure_provider()
        self.configure_sources()
        self.configure_models()
        self.configure_bounds()
        self.configure_resolutions()
        self.configure_grid()
        self.configure_legend()
        self.configure_cache()
        self.configure_metadata()
        self.configure_templates()
        self.configure_search()

    def configure_provider(self):
        self.provider = gws.base.database.provider.get_for(self)
        return True

    def configure_sources(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        self.tableDesc = self.provider.describe(self.tableName)
        if not self.tableDesc:
            raise gws.Error(f'table {self.tableName!r} not found or not readable')
        return True

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.configure_model({}))
        return True

    def configure_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='postgres',
            _defaultProvider=self.provider,
            _defaultTableName=self.tableName
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        b = self.provider.bounds_for_table(self.tableName)
        if b:
            self.bounds = gws.gis.bounds.transform(b, self.defaultBounds.crs)
            return True
        return self.defaultBounds

    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.configure_finder({}))
        return True

    def configure_finder(self, cfg):
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type='postgres',
            _defaultProvider=self.provider,
            _defaultTableName=self.tableName
        )

    def props(self, user):
        return gws.merge(super().props(user), geometryType=self.tableDesc.geometryType)
