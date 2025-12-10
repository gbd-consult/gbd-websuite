from typing import Optional
import gws
import gws.plugin.qgis.provider


class ProjectConfig(gws.ConfigWithAccess):
    title: str = ''
    """Project title."""
    provider: gws.plugin.qgis.provider.Config
    """QGis provider settings."""
    models: Optional[list[gws.ext.config.model]]
    """Data models."""
    mapCacheLifeTime: gws.Duration = '0'
    """Cache life time for base map layers."""


class QfcProject(gws.Node):
    title: str
    qgisProvider: gws.plugin.qgis.provider.Object
    models: list[gws.DatabaseModel]
    mapCacheLifeTime: int

    def configure(self):
        self.title = self.cfg('title', '') or self.uid
        self.qgisProvider = self.create_child(gws.plugin.qgis.provider.Object, self.cfg('provider'))
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))
        self.mapCacheLifeTime = self.cfg('mapCacheLifeTime') or 0


