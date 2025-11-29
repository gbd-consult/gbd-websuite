from typing import Optional
import gws


class Config(gws.Config):
    """Auth method config."""

    secure: bool = True
    """Use only with SSL."""
    allowInsecureFrom: Optional[list[str]]
    """Allow insecure access from these IPs."""


class Object(gws.AuthMethod):
    def configure(self):
        self.secure = self.cfg('secure')
        self.allowInsecureFrom = self.cfg('allowInsecureFrom', default=[])
