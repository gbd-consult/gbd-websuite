import gws
import gws.types as t

from .. import provider, user


@gws.ext.Object('auth.provider.system')
class Object(provider.Object):
    users: t.Dict[str, gws.IUser]

    def configure(self):
        self.users = {
            'guest': user.Guest().init_from_source(self, 'guest'),
            'system': user.System().init_from_source(self, 'system'),
        }

    def authenticate(self, method, credentials):
        # system and guest cannot log in
        return None

    def get_user(self, local_uid):
        return self.users.get(local_uid)
