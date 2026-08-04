"""Base session manager."""

from typing import Optional

import gws
import gws.lib.datetimex


class Config(gws.Config):
    """Configuration for the session manager."""

    lifeTime: gws.Duration = '20m'
    """Session life time, counted from the last request."""
    maxLifeTime: Optional[gws.Duration]
    """Absolute session life time. (added in 8.4)"""


class Object(gws.AuthSessionManager):
    """Base session manager."""

    def configure(self):
        self.lifeTime = self.cfg('lifeTime', default=gws.lib.datetimex.parse_duration(Config.lifeTime))

        p = self.cfg('maxLifeTime') or 0
        if 0 < p < self.lifeTime:
            raise gws.ConfigurationError(f'invalid maxLifeTime={p}, must not be less than lifeTime={self.lifeTime}')
        self.maxLifeTime = p
