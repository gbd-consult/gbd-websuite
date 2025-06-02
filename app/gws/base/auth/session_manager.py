"""Base session manager."""

import gws


class Config(gws.Config):
    """Configuration for the session manager."""
    
    lifeTime: gws.Duration = '1200'
    """Session life time."""


class Object(gws.AuthSessionManager):
    """Base session manager."""

    def configure(self):
        self.lifeTime = self.cfg('lifeTime', default=int(Config.lifeTime))
