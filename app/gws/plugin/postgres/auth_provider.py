"""Provider for the postgres-based authorization.
"""

from typing import Optional, cast

import re

import gws
import gws.base.auth
import gws.base.database
import gws.base.auth.sql_provider
import gws.config.util

gws.ext.new.authProvider('postgres')


class Config(gws.base.auth.sql_provider.Config):
    """Postgres authorization provider (added in 8.1)"""
    pass


class Object(gws.base.auth.sql_provider.Object):
    pass
