"""Mock authorization objects."""

import gws
import gws.base.auth
import gws.lib.jsonx

METHOD_1 = 'mockAuthMethod1'
METHOD_2 = 'mockAuthMethod2'
PROVIDER_1 = 'mockAuthProvider1'
MFA_1 = 'mockAuthMfaAdapter1'

MFA_VALID_CODE = 'yes'

_USER_DATA = {}


def add_user(name, password='', roles=None, token='', **kwargs):
    _USER_DATA[name] = {
        'localUid': name,
        'loginName': name,
        'password': password or '',
        'roles': roles or [],
        'token': token or '',
        **kwargs,
    }


def delete_user(name):
    _USER_DATA.pop(name, None)


def drop_users():
    _USER_DATA.clear()


def system_user():
    return gws.base.auth.user.SystemUser(None, roles=[])


class Method1(gws.base.auth.method.Object):
    pass


class Method2(gws.base.auth.method.Object):
    pass


class Provider1(gws.base.auth.provider.Object):
    def authenticate(self, method, credentials):
        for ud in _USER_DATA.values():
            if credentials.get('username', '') == ud['loginName'] and credentials.get('password', '') == ud['password']:
                return self.get_user(ud['localUid'])
            if ud['token'] and credentials.get('token', '') == ud['token']:
                return self.get_user(ud['localUid'])

    def get_user(self, local_uid):
        for ud in _USER_DATA.values():
            if ud['localUid'] == local_uid:
                return gws.base.auth.user.from_record(self, ud)

    def unserialize_user(self, data):
        d = gws.lib.jsonx.from_string(data)
        _, local_uid = gws.u.split_uid(d['uid'])
        return self.get_user(local_uid)


class MfaAdapter1(gws.base.auth.mfa.Object):
    def verify(self, mfa, payload):
        ok = payload['code'] == MFA_VALID_CODE
        return self.verify_attempt(mfa, ok)


##


def register(specs: gws.SpecRuntime):
    specs.register_object(gws.ext.object.authMethod, METHOD_1, Method1)
    specs.register_object(gws.ext.object.authMethod, METHOD_2, Method2)
    specs.register_object(gws.ext.object.authProvider, PROVIDER_1, Provider1)
    specs.register_object(gws.ext.object.authMultiFactorAdapter, MFA_1, MfaAdapter1)
    return specs
