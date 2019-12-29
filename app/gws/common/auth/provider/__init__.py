import gws.tools.json2
import gws.config
import gws.types as t
import gws.auth.user


#:stub AuthProviderObject
class Object(gws.Object):
    def get_user(self, user_uid: str) -> t.User:
        pass

    def unmarshal_user(self, user_uid: str, json: str) -> t.User:
        s = gws.tools.json2.from_string(json)
        return gws.auth.user.ValidUser().init_from_cache(self, user_uid, s['roles'], s['attributes'])

    def marshal_user(self, user: t.User) -> str:
        return gws.tools.json2.to_string({
            'roles': list(u.roles),
            'attributes': u.attributes
        })
