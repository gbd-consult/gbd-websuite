import gws

from . import user as user_module


class Object(gws.AuthProvider):
    users: dict
    type = 'system'

    def configure(self):
        self.uid = 'gws.base.auth.provider.system'
        self.allowedMethods = []

        g = user_module.GuestUser(self, roles=[gws.c.ROLE_GUEST, gws.c.ROLE_ALL])
        g.attributes = {}
        g.displayName = ''
        g.localUid = 'guest'
        g.loginName = ''
        g.uid = gws.u.join_uid(self.uid, g.localUid)

        s = user_module.SystemUser(self, roles=[])
        s.attributes = {}
        s.displayName = ''
        s.localUid = 'system'
        s.loginName = ''
        s.uid = gws.u.join_uid(self.uid, s.localUid)

        self.users = {'guest': g, 'system': s}

    def authenticate(self, method, credentials):
        # system and guest cannot log in
        return None

    def serialize_user(self, user):
        if user is self.users['guest']:
            return 'guest'
        if user is self.users['system']:
            return 'system'
        raise gws.Error(f'wrong user for system.serialize: {user.uid!r}')

    def unserialize_user(self, data):
        if data in self.users:
            return self.users[data]
        raise gws.Error(f'wrong data for system.unserialize: {data!r}')

    def get_user(self, local_uid):
        return self.users[local_uid]
