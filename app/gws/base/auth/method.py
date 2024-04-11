import gws
import gws.types as t


class Config(gws.Config):
    """Auth method config."""

    secure: bool = True
    """use only with SSL"""


class Object(gws.Node, gws.IAuthMethod):
    def configure(self):
        self.secure = self.cfg('secure')
