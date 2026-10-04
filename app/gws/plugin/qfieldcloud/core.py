"""QField project configuration and object."""

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
    """Project offered to QField.

    Holds the QGIS provider of the source project, the models for editable
    layers and the base map settings used when the project is packaged.
    """

    title: str
    """Project title, the uid if no title is configured."""
    qgisProvider: gws.plugin.qgis.provider.Object
    """QGIS provider for the source project."""
    models: list[gws.DatabaseModel]
    """Configured models for editable layers."""
    mapCacheLifeTime: int
    """How long rendered base maps are reused, in seconds. ``0`` disables the cache."""
    thumbnail: str
    """Path to the thumbnail image, or an empty string."""

    def configure(self):
        self.title = self.cfg('title', '') or self.uid
        self.qgisProvider = self.create_child(gws.plugin.qgis.provider.Object, self.cfg('provider'))
        self.root.app.register_supported_crs(self.qgisProvider.forceCrs)
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))
        self.mapCacheLifeTime = self.cfg('mapCacheLifeTime') or 0
        self.thumbnail = self.cfg('thumbnail') or ''


