"""Provider for the postgres-based authorization."""

import gws
import gws.base.database
import gws.base.database.auth_provider

gws.ext.new.authProvider('postgres')


class Config(gws.base.database.auth_provider.Config):
    """Postgres authorization provider."""

    pass


class Object(gws.base.database.auth_provider.Object):
    pass
