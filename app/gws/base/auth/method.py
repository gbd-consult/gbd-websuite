from typing import Optional
import gws


class Config(gws.Config):
    """Common options for authentication methods."""

    secure: bool = True
    """Accept credentials only over HTTPS."""
    allowInsecureFrom: Optional[list[str]]
    """IP addresses allowed to use this method without HTTPS."""


class Object(gws.AuthMethod):
    def configure(self):
        self.secure = self.cfg('secure')
        self.allowInsecureFrom = self.cfg('allowInsecureFrom', default=[])
