"""PostgreSQL model."""

import gws
import gws.base.database.model
import gws.base.feature

from . import provider


@gws.ext.config.model('postgres')
class Config(gws.base.database.model.Config):
    """Data model for a PostgreSQL table."""

    pass


@gws.ext.props.model('postgres')
class Props(gws.base.database.model.Props):
    pass


@gws.ext.object.model('postgres')
class Object(gws.base.database.model.Object):
    """Data model for the records of a PostgreSQL table."""

    pass
