"""Provider for the postgres-based authorization.
"""

import re

import gws
import gws.base.auth
import gws.base.database
import gws.base.auth.sql_provider
import gws.types as t

gws.ext.new.authProvider('postgres')


class Config(gws.base.auth.sql_provider.Config):
    pass


class Object(gws.base.auth.sql_provider.Object):
    def configure(self):
        self.dbProvider = t.cast(gws.DatabaseProvider, gws.base.database.provider.get_for(self))
