import gws.tools.json2
import gws.config
import gws.types as t
import gws.auth.user


class Object(gws.Object, t.AuthProviderObject):
    def unmarshal_user(self, user_uid, s):
        s = gws.tools.json2.from_string(s)
        return gws.auth.user.ValidUser().init_from_cache(self, user_uid, s['roles'], s['attributes'])

    def marshal_user(self, u):
        return gws.tools.json2.to_string({
            'roles': list(u.roles),
            'attributes': u.attributes
        })
