"""Base session manager."""

import gws
import gws.types as t


class Config(gws.Config):
    lifeTime: gws.Duration = '1200'
    """session life time"""


class Object(gws.Node, gws.IAuthSessionManager):
    """Base session manager."""

    def configure(self):
        self.authMgr = t.cast(gws.IAuthManager, self.cfg('_defaultManager'))
        self.lifeTime = self.cfg('lifeTime', default=int(Config.lifeTime))
