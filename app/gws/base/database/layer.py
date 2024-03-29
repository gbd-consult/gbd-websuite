import gws
import gws.base.database
import gws.base.layer
import gws.gis.bounds
import gws.base.feature
import gws.gis.crs
import gws.base.shape
import gws.config.util
import gws.types as t


class Config(gws.base.layer.Config):
    """Database layer"""

    dbUid: t.Optional[str]
    """Database provider uid"""
    tableName: str
    """sql table name"""


class Object(gws.base.layer.vector.Object):
    provider: gws.IDatabaseProvider
    tableName: str

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        self.provider = gws.base.database.provider.get_for(self)
        return True

    def configure_sources(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        desc = self.provider.describe(self.tableName)
        if not desc:
            raise gws.Error(f'table {self.tableName!r} not found or not readable')
        self.geometryType = desc.geometryType
        self.geometryCrs = gws.gis.crs.get(desc.geometrySrid)
        return True

    def configure_models(self):
        return gws.config.util.configure_models(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider,
            _defaultTableName=self.tableName
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        b = self.provider.table_bounds(self.tableName)
        if b:
            self.bounds = gws.gis.bounds.transform(b, self.mapCrs)
            return True

    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.create_finder(None))
        return True

    def create_finder(self, cfg):
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider,
            _defaultTableName=self.tableName
        )
