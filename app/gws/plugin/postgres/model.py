"""Postgres models."""

import gws
import gws.base.database.model
import gws.base.feature

from . import provider

gws.ext.new.model('postgres')


class Config(gws.base.database.model.Config):
    """Postgres database model configuration."""

    pass


class Props(gws.base.database.model.Props):
    pass


class Object(gws.base.database.model.Object):
    pass
