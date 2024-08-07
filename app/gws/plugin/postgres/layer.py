import gws
import gws.base.database.layer

from . import provider

gws.ext.new.layer('postgres')


class Config(gws.base.database.layer.Config):
    """Postgres layer"""
    pass


class Object(gws.base.database.layer.Object):
    db: provider.Object
