"""Mock objects for testing."""

import gws
import gws.base.auth
import gws.base.web.wsgi
import gws.lib.net
import gws.lib.jsonx
import gws.base.auth.user as user_api


class AuthMethod1(gws.base.auth.method.Object):
    pass


class AuthMethod2(gws.base.auth.method.Object):
    pass


_USER_DATA = {}


def add_user(name, password='', roles=None, **kwargs):
    _USER_DATA[name] = {
        'localUid': name,
        'loginName': name,
        'password': password or '',
        'roles': roles or [],
        **kwargs
    }


def delete_user(name):
    _USER_DATA.pop(name, None)


class AuthProvider1(gws.base.auth.provider.Object):
    def authenticate(self, method, credentials):
        for ud in _USER_DATA.values():
            if credentials.get('username', '') == ud['loginName'] and credentials.get('password', '') == ud['password']:
                return self.get_user(ud['localUid'])

    def get_user(self, local_uid):
        for ud in _USER_DATA.values():
            if ud['localUid'] == local_uid:
                return gws.base.auth.user.from_record(self, ud)

    def unserialize_user(self, data):
        d = gws.lib.jsonx.from_string(data)
        _, local_uid = gws.u.split_uid(d['uid'])
        return self.get_user(local_uid)


class AuthMfaAdapter1(gws.base.auth.mfa.Object):
    VALID_CODE = 'yes'

    def verify(self, mfa, payload):
        ok = payload['code'] == self.VALID_CODE
        return self.verify_attempt(mfa, ok)


##

def register(specs: gws.SpecRuntime):
    specs.register_object(gws.ext.object.authMethod, 'mockAuthMethod1', AuthMethod1)
    specs.register_object(gws.ext.object.authMethod, 'mockAuthMethod2', AuthMethod2)
    specs.register_object(gws.ext.object.authProvider, 'mockAuthProvider1', AuthProvider1)
    specs.register_object(gws.ext.object.authMultiFactorAdapter, 'mockAuthMfaAdapter1', AuthMfaAdapter1)
    return specs
