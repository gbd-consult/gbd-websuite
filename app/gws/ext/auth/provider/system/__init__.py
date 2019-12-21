import gws
import gws.common.auth.provider
import gws.auth.error
import gws.auth.user

import gws.types as t

class Object(gws.common.auth.provider.Object):

    def authenticate_user(self, login, password, **kw):
        # system and guest cannot log in
        return None

    def get_user(self, user_uid):
        user: t.AuthUser = None
        if user_uid == 'guest':
            user = self.root.create(gws.auth.user.Guest)
            return user.init_from_source(self, uid='guest')
        if user_uid == 'system':
            user = self.root.create(gws.auth.user.System)
            return user.init_from_source(self, uid='system')

    def unmarshal_user(self, user_uid, s=''):
        return self.get_user(user_uid)

    def marshal_user(self, user):
        return ''
