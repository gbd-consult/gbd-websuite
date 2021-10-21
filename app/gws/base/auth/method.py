import gws
from . import error


class Config(gws.Config):
    """Auth method config."""

    secure: bool = True  #: use only with SSL


class Object(gws.Node, gws.IAuthMethod):
    def configure(self):
        self.secure = self.var('secure')

    def login(self, auth, credentials, req):
        raise error.AccessDenied()
