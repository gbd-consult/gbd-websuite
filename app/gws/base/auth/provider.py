import gws
import gws.types as t
import gws.lib.json2
from . import user as user_api


class Config(gws.Config):
    """Auth provider config."""

    allowedMethods: t.Optional[t.List[str]]  #: allowed authorization methods


class Object(gws.Object, gws.IAuthProvider):
    allowed_methods: t.List[str]

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
        return user_api.AuthorizedUser().init_from_data(self, d['local_uid'], d['roles'], d['attributes'])
