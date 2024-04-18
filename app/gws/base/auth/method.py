import gws


class Config(gws.Config):
    """Auth method config."""

    secure: bool = True
    """use only with SSL"""


class Object(gws.AuthMethod):
    def configure(self):
        self.secure = self.cfg('secure')
