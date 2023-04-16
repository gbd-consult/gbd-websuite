import gws
import gws.base.database.layer
import gws.types as t

from . import provider

gws.ext.new.layer('postgres')


class Config(gws.base.database.layer.Config):
    """Postgres layer"""
    pass


class Object(gws.base.database.layer.Object):
    provider: provider.Object
