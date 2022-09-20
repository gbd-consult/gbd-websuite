import gws
import gws.lib.json2
import gws.types as t

from . import user as user_api


class Config(gws.Config):
    """Auth provider config."""

    allowedMethods: t.Optional[t.List[str]]  #: allowed authorization methods


class Object(gws.Node, gws.IAuthProvider):
    def configure(self):
        self.allowedMethods = self.var('allowedMethods', default=[])

    def authenticate(self, method, credentials):
        return None

    def serialize_user(self, user):
        return gws.lib.json2.to_string(user_api.to_dict(user))

    def unserialize_user(self, data):
        d = gws.lib.json2.from_string(data)
        return user_api.from_dict(user_api.AuthorizedUser, self, d) if d else None
