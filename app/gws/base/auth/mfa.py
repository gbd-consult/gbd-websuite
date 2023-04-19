import gws
import gws.types as t


class Config(gws.Config):
    """MFA method config."""

    pass


class Object(gws.Node, gws.IAuthMfa):
    def configure(self):
        self.authMgr = t.cast(gws.IAuthManager, self.cfg('_defaultManager'))
