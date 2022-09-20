import gws

from . import error


class Config(gws.Config):
    """MFA method config."""

    pass


class Object(gws.Node, gws.IAuthMfa):
    pass
