import gws.config
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


def get_user(user_fid):
    provider_uid, user_uid = user_fid.split('//', 1)
    prov: t.IAuthProvider = gws.config.root().find('gws.ext.auth.provider', provider_uid)
    return prov.get_user(user_uid)


def role(name):
    return Role(name)
