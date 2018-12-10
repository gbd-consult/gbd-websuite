import gws
import gws.auth.provider
import gws.auth.error
import gws.auth.user


class Object(gws.auth.provider.Object):

    def authenticate_user(self, login, password, **kw):
        # system and guest cannot log in
        return None

    def get_user(self, user_uid):
        if user_uid == 'guest':
            return self.root.create(gws.auth.user.Guest).init_from_source(self, uid='guest')
        if user_uid == 'system':
            return self.root.create(gws.auth.user.System).init_from_source(self, uid='system')

    def unmarshal_user(self, user_uid, s=''):
        return self.get_user(user_uid)

    def marshal_user(self, user):
        return ''
