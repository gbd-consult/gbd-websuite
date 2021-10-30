import gws
import gws.lib.json2
import gws.types as t

from . import user as user_api


class Config(gws.Config):
    """Auth provider config."""

    allowedMethods: t.Optional[t.List[str]]  #: allowed authorization methods


class Object(gws.Node, gws.IAuthProvider):
    def configure(self):
        self.allowed_methods = self.var('allowedMethods', default=[])

    def authenticate(self, method, credentials):
        return None

    def serialize_user(self, user):
        return gws.lib.json2.to_string({
            'local_uid': user.local_uid,
            'roles': list(user.roles),
            'attributes': user.attributes
        })

    def unserialize_user(self, ser):
        d = gws.lib.json2.from_string(ser)
        return user_api.create(user_api.AuthorizedUser, self, d['local_uid'], set(d['roles']), d['attributes'])
