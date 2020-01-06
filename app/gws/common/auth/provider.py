import gws.tools.json2
import gws.config
import gws.types as t

from . import user


#:export IAuthProvider
class Object(gws.Object, t.IAuthProvider):
    def get_user(self, user_uid: str) -> t.IUser:
        pass

    def authenticate(self, login: str, password: str, **kwargs) -> t.IUser:
        pass

    def user_from_dict(self, d: dict) -> t.IUser:
        return user.ValidUser().init_from_props(self, d['user_uid'], d['roles'], d['attributes'])

    def user_to_dict(self, u: t.IUser) -> dict:
        return {
            'provider_uid': self.uid,
            'user_uid': u.uid,
            'roles': list(u.roles),
            'attributes': u.attributes
        }
