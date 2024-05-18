"""Provider for the postgres-based authorization.
"""

from typing import Optional, cast

import re

import gws
import gws.base.auth
import gws.base.database
import gws.base.auth.sql_provider

gws.ext.new.authProvider('postgres')


class Config(gws.base.auth.sql_provider.Config):
    """Postgres authorization provider (added in 8.1)"""
    pass


class Object(gws.base.auth.sql_provider.Object):
    def configure(self):
        self.dbProvider = cast(gws.DatabaseProvider, gws.base.database.provider.get_for(self))
