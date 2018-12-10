import gws.config
import gws.types as t
from .error import *
import gws.auth.stores.sqlite
from . import user

User = user.User


def init():
    gws.auth.stores.sqlite.init()


def authenticate_user(login, password, **kw):
    for prov in gws.config.find_all('gws.ext.auth.provider'):
        gws.log.info('trying provider %r for login %r' % (prov.uid, login))
        usr = prov.authenticate_user(login, password, **kw)
        if usr:
            return usr


def get_user(user_fid):
    provider_uid, user_uid = user_fid.split('//', 1)
    prov: t.AuthProviderInterface = gws.config.find('gws.ext.auth.provider', provider_uid)
    return prov.get_user(user_uid)


def role(name):
    return user.Role(name)
