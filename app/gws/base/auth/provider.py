import gws
import gws.lib.jsonx
import gws.types as t

from . import user as user_api


class Config(gws.Config):
    """Auth provider config."""

    allowedMethods: t.Optional[list[str]]
    """allowed authorization methods"""


class Object(gws.AuthProvider):
    def configure(self):
        self.allowedMethods = self.cfg('allowedMethods', default=[])

    def authenticate(self, method, credentials):
        return None

    def serialize_user(self, user):
        return gws.lib.jsonx.to_string(user_api.to_dict(user))

    def unserialize_user(self, data):
        d = gws.lib.jsonx.from_string(data)
        return user_api.from_dict(self, d) if d else None
