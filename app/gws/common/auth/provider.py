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

    def unmarshal_user(self, user_uid: str, json: str) -> t.IUser:
        s = gws.tools.json2.from_string(json)
        return user.ValidUser().init_from_cache(self, user_uid, s['roles'], s['attributes'])

    def marshal_user(self, u: t.IUser) -> str:
        return gws.tools.json2.to_string({
            'roles': list(u.roles),
            'attributes': u.attributes
        })
