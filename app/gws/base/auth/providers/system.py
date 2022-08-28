import gws
import gws.types as t

from .. import provider, user


@gws.ext.object.authProvider('system')
class Object(provider.Object):
    users: t.Dict[str, user.User]

    def configure(self):
        super().configure()
        self.uid = 'gws.base.auth.providers.system'
        self.users = {
            'guest': user.create(user.Guest, self, 'guest', [gws.ROLE_GUEST]),
            'system': user.create(user.System, self, 'system', []),
        }

    def authenticate(self, method, credentials):
        # system and guest cannot log in
        return None

    def get_user(self, local_uid):
        return self.users.get(local_uid)
