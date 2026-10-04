"""PostgreSQL layer."""

import gws
import gws.base.database.layer

from . import provider


@gws.ext.config.layer('postgres')
class Config(gws.base.database.layer.Config):
    """Layer that shows features from a PostgreSQL table."""
    pass


@gws.ext.object.layer('postgres')
class Object(gws.base.database.layer.Object):
    """Vector layer that shows features from a PostgreSQL table."""

    db: provider.Object
    """PostgreSQL database provider."""
