import gws
import gws.common.auth.provider
import gws.common.auth.user

import gws.types as t


class Object(gws.common.auth.provider.Object):
    guest_user: t.IUser = None
    system_user: t.IUser = None

    def authenticate(self, method: t.IAuthMethod, login, password, **kw):
        # system and guest cannot log in
        return None

    def get_user(self, user_uid):
        if user_uid == 'guest':
            if not self.guest_user:
                self.guest_user = self.root.create(gws.common.auth.user.Guest)
                self.guest_user.init_from_source(self, uid='guest')
            return self.guest_user

        if user_uid == 'system':
            if not self.system_user:
                self.system_user = self.root.create(gws.common.auth.user.System)
                self.system_user.init_from_source(self, uid='system')
            return self.system_user
