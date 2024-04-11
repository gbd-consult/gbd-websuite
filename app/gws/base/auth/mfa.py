import gws
import gws.types as t


class Config(gws.Config):
    """MFA method config."""

    pass


class Object(gws.Node, gws.IAuthMfa):
    pass
