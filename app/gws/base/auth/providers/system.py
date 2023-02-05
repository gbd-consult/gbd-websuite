import gws
import gws.types as t

from .. import provider, user


gws.ext.new.authProvider('system')

class Object(provider.Object):
    users: t.Dict[str, user.User]

    def configure(self):
        self.uid = 'gws.base.auth.providers.system'
        self.users = {
            'guest': user.from_args(user.Guest, provider=self, localUid='guest', roles=[gws.ROLE_GUEST]),
            'system': user.from_args(user.System, provider=self, localUid='system'),
        }

    def authenticate(self, method, credentials):
        # system and guest cannot log in
        return None

    def get_user(self, local_uid):
        return self.users.get(local_uid)
