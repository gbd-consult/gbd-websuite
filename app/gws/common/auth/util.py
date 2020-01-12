import gws.config
import gws.tools.json2

import gws.types as t

from . import user
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


def get_user(user_fid):
    # see user.fid
    provider_uid, user_uid = user.parse_fid(user_fid)
    prov: t.IAuthProvider = gws.config.root().find('gws.ext.auth.provider', provider_uid)
    return prov.get_user(user_uid)


def serialize_user(u: t.IUser) -> str:
    return gws.tools.json2.to_string(u.provider.user_to_dict(u))


def unserialize_user(s: str) -> t.IUser:
    d = gws.tools.json2.from_string(s)
    prov: t.IAuthProvider = gws.config.root().find('gws.ext.auth.provider', d['provider_uid'])
    return prov.user_from_dict(d)


def role(name):
    return user.Role(name)
