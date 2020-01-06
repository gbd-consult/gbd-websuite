import gws
import gws.common.auth.provider
import gws.common.auth.user

import gws.types as t


class Object(gws.common.auth.provider.Object):

    def authenticate(self, login, password, **kw):
        # system and guest cannot log in
        return None

    def get_user(self, user_uid):
        user: t.IUser = None
        if user_uid == 'guest':
            user = self.root.create(gws.common.auth.user.Guest)
            return user.init_from_source(self, uid='guest')
        if user_uid == 'system':
            user = self.root.create(gws.common.auth.user.System)
            return user.init_from_source(self, uid='system')

