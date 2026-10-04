"""Base authentication method."""

from typing import Optional
import gws


class Config(gws.Config):
    """Common options for authentication methods."""

    secure: bool = True
    """Accept credentials only over HTTPS."""
    allowInsecureFrom: Optional[list[str]]
    """IP addresses allowed to use this method without HTTPS."""


class Object(gws.AuthMethod):
    """Base authentication method.

    Reads the ``secure`` and ``allowInsecureFrom`` options, which the manager
    checks before it uses the method. Subclasses implement opening and closing
    sessions for web requests.
    """

    def configure(self):
        self.secure = self.cfg('secure')
        self.allowInsecureFrom = self.cfg('allowInsecureFrom', default=[])
