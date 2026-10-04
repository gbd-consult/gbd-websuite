"""Base layer for database tables."""

from typing import Optional

import gws
import gws.base.database
import gws.base.layer
import gws.lib.bounds
import gws.base.feature
import gws.lib.crs
import gws.lib.shape
import gws.config.util


class Config(gws.base.layer.Config):
    """Database layer."""

    dbUid: Optional[str]
    """UID of the database provider."""
    tableName: str
    """Table name, optionally schema-qualified."""


class Object(gws.base.layer.vector.Object):
    """Base vector layer for a database table.

    Shows the features of one table. Provides subclasses with the database
    provider, the table description and default models and finders of the
    layer's type for the table.
    """

    db: gws.DatabaseProvider
    """Database provider."""
    tableName: str
    """Table name, optionally schema-qualified."""

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        return gws.config.util.configure_database_provider_for(self)

    def configure_sources(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        desc = self.db.describe(self.tableName)
        if not desc:
            raise gws.Error(f'table not found or not readable: {self.tableName!r}')
        self.geometryType = desc.geometryType
        self.geometryCrs = gws.lib.crs.get(desc.geometrySrid)
        return True

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a model of the layer's type for the layer's table.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model object.
        """
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultDb=self.db,
            _defaultTableName=self.tableName
        )

    def configure_extent(self):
        if super().configure_extent():
            return True
        b = self.db.table_bounds(self.tableName)
        if b:
            ext = gws.lib.bounds.wgs_extent(b, pad=True)
            if ext:
                self.wgsExtent = ext
                return True

    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.create_finder(None))
        return True

    def create_finder(self, cfg):
        """Create a finder of the layer's type for the layer's table.

        Args:
            cfg: Finder configuration, or ``None`` for the default finder.

        Returns:
            The finder object.
        """
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type=self.extType,
            _defaultDb=self.db,
            _defaultTableName=self.tableName
        )
