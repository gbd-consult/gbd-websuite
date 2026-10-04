"""PostgreSQL finder."""

from typing import Optional, cast

import gws
import gws.base.database
import gws.base.model
import gws.base.search
import gws.config.util


@gws.ext.config.finder('postgres')
class Config(gws.base.search.finder.Config):
    """Search in a PostgreSQL table."""

    dbUid: Optional[str]
    """UID of the database provider."""
    tableName: str
    """Table to search, optionally schema-qualified."""
    sqlFilter: Optional[str]
    """SQL condition added to all search queries."""


@gws.ext.object.finder('postgres')
class Object(gws.base.search.finder.Object):
    """Finder that searches a PostgreSQL table through its ``postgres`` models."""

    db: gws.DatabaseProvider
    """Database provider."""
    tableName: str
    """Table to search."""

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
        """Set the database provider from ``dbUid``, or the first ``postgres`` provider.

        Returns:
            ``True`` if a provider was set.

        Raises:
            ``gws.Error``: If no provider is found.
        """
        return gws.config.util.configure_database_provider_for(self)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a ``postgres`` model for the table of the finder.

        The model gets the database provider, the table and the ``sqlFilter`` of the finder.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model object.
        """
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            sqlFilter=self.cfg('sqlFilter'),
            _defaultDb=self.db,
            _defaultTableName=self.tableName
        )
