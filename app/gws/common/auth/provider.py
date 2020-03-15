import gws.config

import gws.types as t

from . import user


class Config(t.WithType):
    """Auth provider config."""

    allowedMethods: t.Optional[t.List[str]]


#:export IAuthProvider
class Object(gws.Object, t.IAuthProvider):
    allowed_methods: t.List[str]

    def configure(self):
        super().configure()
        self.allowed_methods = self.var('allowedMethods')

    def get_user(self, user_uid: str) -> t.Optional[t.IUser]:
        pass

    def authenticate(self, method: t.IAuthMethod, login: str, password: str, **kwargs) -> t.Optional[t.IUser]:
        pass

    def user_from_dict(self, d: dict) -> t.IUser:
        return user.ValidUser().init_from_data(self, d['user_uid'], d['roles'], d['attributes'])

    def user_to_dict(self, u: t.IUser) -> dict:
        return {
            'provider_uid': self.uid,
            'user_uid': u.uid,
            'roles': list(u.roles),
            'attributes': u.attributes
        }
