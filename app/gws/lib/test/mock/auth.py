import gws.base.auth.provider
import gws.base.auth.user


class Provider(gws.base.auth.provider.Object):
    def __init__(self, users):
        super().__init__()
        self.users = users

    def authenticate(self, method, credentials):
        for login, roles in self.users.items():
            if credentials.get('login') == login:
                return self.get_user(login)

    def get_user(self, local_uid):
        for login, roles in self.users.items():
            if local_uid == login:
                return gws.base.auth.user.from_args(
                    gws.base.auth.user.AuthorizedUser,
                    provider=self,
                    roles=roles,
                    localUid=local_uid,
                )
