import gws
import gws.types as t


from .. import core


@gws.ext.Object('auth.provider.system')
class Object(core.Provider):
    guest_user: core.Guest = t.cast(core.Guest, None)
    system_user: core.System = t.cast(core.System, None)

    def authenticate(self, method, credentials):
        # system and guest cannot log in
        return None

    def get_user(self, user_uid):
        if user_uid == 'guest':
            if not self.guest_user:
                self.guest_user = core.Guest().init_from_source(self, 'guest')
            return self.guest_user

        if user_uid == 'system':
            if not self.system_user:
                self.system_user = core.System().init_from_source(self, 'system')
            return self.system_user
