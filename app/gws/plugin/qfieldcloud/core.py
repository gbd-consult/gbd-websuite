"""QField project configuration and object."""

from typing import Optional
import gws
import gws.plugin.qgis.provider
import gws.plugin.model_field.file as file_field

from . import api


class DeltaApplyError(gws.Error):
    """Raised when a delta payload cannot be applied."""


class DeltaStatus(gws.Data):
    """Status of a delta, as returned to QField."""

    id: str
    """Delta id."""
    deltafile_id: str
    """Id of the delta payload."""
    created_by: str
    """Login name of the user who sent the delta."""
    created_at: str
    """Creation time."""
    updated_at: str
    """Last update time."""
    status: api.DeltaStatusType
    """Delta status."""
    output: str
    """Error message, if the delta could not be applied."""
    content: api.Delta
    """Delta as sent by QField."""


class SyncFileConfig(gws.Config):
    """File field whose content is sent to QField. (added in 8.5)"""

    field: str
    """Name of a file field in the project models."""
    useBlob: bool = False
    """Send the content as a binary GeoPackage column instead of a file."""


class SyncFile(gws.Data):
    """File field whose content is sent to QField."""

    model: gws.Model
    """Model the file field belongs to."""
    field: file_field.Object
    """File field."""
    useBlob: bool
    """Send the content as a binary GeoPackage column, named after the ``contentColumn``, instead of a file under the path in the ``nameColumn``."""


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
    syncFiles: Optional[list[SyncFileConfig]]
    """File fields whose content is sent to QField. (added in 8.5)"""


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
    syncFiles: list[SyncFile]
    """File fields whose content is sent to QField."""

    def configure(self):
        self.title = self.cfg('title', '') or self.uid
        self.qgisProvider = self.create_child(gws.plugin.qgis.provider.Object, self.cfg('provider'))
        self.root.app.register_supported_crs(self.qgisProvider.forceCrs)
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))
        self.mapCacheLifeTime = self.cfg('mapCacheLifeTime') or 0
        self.thumbnail = self.cfg('thumbnail') or ''
        self.syncFiles = [self.configure_sync_file(c) for c in self.cfg('syncFiles', default=[])]

    def configure_sync_file(self, cfg: SyncFileConfig) -> SyncFile:
        """Find the file field for a ``syncFiles`` entry.

        The columns are checked in the field config, because file fields set their column names in ``post_configure``.

        Args:
            cfg: ``syncFiles`` entry.

        Returns:
            The file field and the blob flag.

        Raises:
            ``gws.ConfigurationError``: If the name does not match exactly one file field in the project models,
                the field has no ``contentColumn``, or has no ``nameColumn`` and ``useBlob`` is not set.
        """
        found = [f for m in self.models for f in m.fields if f.name == cfg.field and isinstance(f, file_field.Object)]
        if len(found) != 1:
            raise gws.ConfigurationError(f'syncFiles: expected one file field {cfg.field!r}, found {len(found)}')
        fld = found[0]
        return SyncFile(
            model=fld.model,
            field=fld,
            useBlob=bool(cfg.useBlob),
        )


def gp_field_name(name: str) -> str:
    """GeoPackage column name for a model field name. ``fid`` is renamed to ``fid_gws``, because GDAL uses it internally."""
    if name.lower() == 'fid':
        return 'fid_gws'
    return name


def db_field_name(name: str) -> str:
    """Model field name for a GeoPackage column name, the reverse of ``gp_field_name``."""
    if name == 'fid_gws':
        return 'fid'
    return name
