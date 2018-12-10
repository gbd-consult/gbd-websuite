import gws.tools.json2 as json2
import gws.config
import gws.types as t
from . import user


class Object(gws.Object, t.AuthProviderInterface):
    def unmarshal_user(self, user_uid, s):
        s = json2.from_string(s)
        return self.root.create(user.ValidUser).init_from_cache(self, user_uid, s['roles'], s['attributes'])

    def marshal_user(self, u):
        return json2.to_string({
            'roles': list(u.roles),
            'attributes': u.attributes
        })

