"""PostgreSQL authorization provider."""

import gws
import gws.base.database
import gws.base.database.auth_provider


@gws.ext.config.authProvider('postgres')
class Config(gws.base.database.auth_provider.Config):
    """Authentication provider that checks users with SQL queries in PostgreSQL."""

    pass


@gws.ext.object.authProvider('postgres')
class Object(gws.base.database.auth_provider.Object):
    """Authorization provider that checks credentials and loads users with SQL queries in PostgreSQL."""

    pass
