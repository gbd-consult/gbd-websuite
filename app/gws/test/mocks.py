import gws
import gws.base.auth


class Method1(gws.base.auth.method.Object):
    pass


class Method2(gws.base.auth.method.Object):
    pass


USERS = [
    dict(
        login='login1',
        password='password1',
        localUid='user1',
        roles=['role1'],
        displayName='USER_1',
    )
]


class Provider1(gws.base.auth.provider.Object):
    def authenticate(self, method, credentials):
        login = credentials.get('login', '')
        passw = credentials.get('password', '')
        for usr in USERS:
            if usr['login'] == login and usr['password'] == passw:
                return self.get_user(usr['localUid'])

    def get_user(self, local_uid):
        for usr in USERS:
            if usr['localUid'] == local_uid:
                return gws.base.auth.user.from_args(self, **usr)


def register(specs: gws.SpecRuntime):
    specs.register_object(gws.ext.object.authMethod, 'method1', Method1)
    specs.register_object(gws.ext.object.authMethod, 'method2', Method2)
    specs.register_object(gws.ext.object.authProvider, 'provider1', Provider1)
    return specs








