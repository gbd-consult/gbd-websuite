"""Postgres models."""

import gws
import gws.base.database.model
import gws.base.feature
import gws.types as t

from . import provider

gws.ext.new.model('postgres')


class Props(gws.base.database.model.Props):
    pass


class Config(gws.base.database.model.Config):
    pass


class Object(gws.base.database.model.Object):
    pass
