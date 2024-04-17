"""Base session manager."""

import gws


class Config(gws.Config):
    lifeTime: gws.Duration = '1200'
    """session life time"""


class Object(gws.AuthSessionManager):
    """Base session manager."""

    def configure(self):
        self.lifeTime = self.cfg('lifeTime', default=int(Config.lifeTime))
