import gws.config
import gws.tools.json2

import gws.types as t

from .user import Role
from .stores import sqlite


def init():
    sqlite.init()


def authenticate(login, password, **kw):
    prov: t.IAuthProvider
    for prov in gws.config.root().find_all('gws.ext.auth.provider'):
        gws.log.info('trying provider %r for login %r' % (prov.uid, login))
        usr = prov.authenticate(login, password, **kw)
        if usr:
            return usr


def get_user(user_full_uid):
    # see user.full_uid
    provider_uid, user_uid = gws.tools.json2.from_string(user_full_uid)
    prov: t.IAuthProvider = gws.config.root().find('gws.ext.auth.provider', provider_uid)
    return prov.get_user(user_uid)


def serialize_user(user: t.IUser) -> str:
    return gws.tools.json2.to_string(user.provider.user_to_dict(user))


def unserialize_user(s: str) -> t.IUser:
    d = gws.tools.json2.from_string(s)
    prov: t.IAuthProvider = gws.config.root().find('gws.ext.auth.provider', d['provider_uid'])
    return prov.user_from_dict(d)


def role(name):
    return Role(name)
