import gws
import gws.base.database.layer

from . import provider


@gws.ext.config.layer('postgres')
class Config(gws.base.database.layer.Config):
    """Postgres layer"""
    pass


@gws.ext.object.layer('postgres')
class Object(gws.base.database.layer.Object):
    db: provider.Object
