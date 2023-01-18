"""Postgres models."""

import gws
import gws.base.database.model
import gws.base.feature
import gws.types as t

from . import provider


@gws.ext.props.model('postgres')
class Props(gws.base.database.model.Props):
    pass


@gws.ext.config.model('postgres')
class Config(gws.base.database.model.Config):
    pass


@gws.ext.object.model('postgres')
class Object(gws.base.database.model.Object):
    pass
