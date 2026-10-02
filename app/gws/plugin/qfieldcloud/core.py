from typing import Optional
import gws
import gws.plugin.qgis.provider


class ProjectConfig(gws.ConfigWithAccess):
    """Project offered to QField, packaged from a QGIS project."""

    title: str = ''
    """Project title shown in QField."""
    provider: gws.plugin.qgis.provider.Config
    """QGIS project the QField package is built from."""
    models: Optional[list[gws.ext.config.model]]
    """Data models for editable layers, matched by table name."""
    mapCacheLifeTime: gws.Duration = '0'
    """How long rendered offline base maps are reused."""
    thumbnail: Optional[gws.FilePath]
    """Thumbnail image shown in the project details. (added in 8.5)"""


class QfcProject(gws.Node):
    title: str
    qgisProvider: gws.plugin.qgis.provider.Object
    models: list[gws.DatabaseModel]
    mapCacheLifeTime: int
    thumbnail: str

    def configure(self):
        self.title = self.cfg('title', '') or self.uid
        self.qgisProvider = self.create_child(gws.plugin.qgis.provider.Object, self.cfg('provider'))
        self.root.app.register_supported_crs(self.qgisProvider.forceCrs)
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))
        self.mapCacheLifeTime = self.cfg('mapCacheLifeTime') or 0
        self.thumbnail = self.cfg('thumbnail') or ''


