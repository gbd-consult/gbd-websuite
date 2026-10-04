"""Base authentication provider."""

from typing import Optional

import gws
import gws.lib.jsonx

from . import user as user_api


class Config(gws.Config):
    """Auth provider config."""

    allowedMethods: Optional[list[str]]
    """Authentication methods this provider accepts."""


class Object(gws.AuthProvider):
    """Base authentication provider.

    Reads the ``allowedMethods`` option and serializes users as JSON of all their
    fields, roles, attributes and data. Subclasses implement authentication and
    user lookup; the default ``authenticate`` returns no user.
    """

    def configure(self):
        self.allowedMethods = self.cfg('allowedMethods', default=[])

    def authenticate(self, method, credentials):
        return None

    def serialize_user(self, user):
        return gws.lib.jsonx.to_string(user_api.to_dict(user))

    def unserialize_user(self, data):
        d = gws.lib.jsonx.from_string(data)
        return user_api.from_dict(self, d) if d else None
