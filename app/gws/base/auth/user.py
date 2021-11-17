import gws
import gws.core.tree
import gws.types as t

from . import error

_DELIM = '___'


def make_uid(user):
    return f'{user.provider.uid}{_DELIM}{user.local_uid}'


def parse_uid(user_uid):
    s = user_uid.split(_DELIM, 1)
    if len(s) == 2:
        return s
    raise gws.Error(f'invalid user uid: {user_uid!r}')


##

class Props(gws.Props):
    displayName: str


# https://tools.ietf.org/html/rfc4519

_aliases = [
    ('c', 'countryName'),
    ('cn', 'commonName'),
    ('dc', 'domainComponent'),
    ('l', 'localityName'),
    ('o', 'organizationName'),
    ('ou', 'organizationalUnitName'),
    ('sn', 'surname'),
    ('st', 'stateOrProvinceName'),
    ('street', 'streetAddress'),

    # non-standard
    ('login', 'userPrincipalName'),
]


class User(gws.IUser):
    def __init__(self, provider, local_uid, roles, attributes):
        super().__init__()
        self.attributes = dict(attributes or {})
        self.provider = provider
        self.roles = set(roles)
        self.roles.add(gws.ROLE_ALL)
        self.local_uid = local_uid
        self.uid = make_uid(self)
        self.is_guest = gws.ROLE_GUEST in self.roles
        self.display_name = str(self.attributes.get('displayName', ''))
        gws.log.debug(f'inited user: prov={provider.uid!r} local_uid={local_uid!r} roles={roles!r}')

    def props_for(self, user):
        return gws.Data(displayName=self.display_name)

    def can_use(self, obj, context=None):
        return gws.core.tree.user_can_use(self, obj, context)

    def require(self, klass=None, uid=None):
        obj = self.provider.root.find(klass, uid)
        if not obj:
            raise error.ObjectNotFound(klass, uid)
        if not self.can_use(obj):
            raise error.AccessDenied(klass, uid)
        return obj

    def require_project(self, uid):
        return t.cast(gws.IProject, self.require('gws.base.project', uid))

    def require_layer(self, uid):
        return t.cast(gws.ILayer, self.require('gws.ext.layer', uid))

    def acquire(self, klass=None, uid=None):
        obj = self.provider.root.find(klass, uid)
        if obj and self.can_use(obj):
            return obj


class Guest(User):
    def __init__(self, provider, local_uid, roles, attributes):
        super().__init__(provider, local_uid, roles, attributes)
        self.roles.add(gws.ROLE_GUEST)
        self.is_guest = True


class System(User):
    def can_use(self, obj, context=None):
        return True


class Nobody(User):
    def can_use(self, obj, context=None):
        return False


class AuthorizedUser(User):
    pass

class Admin(AuthorizedUser):
    def can_use(self, obj, context=None):
        return True


##


def create(cls, provider: gws.IAuthProvider, local_uid: str, roles=None, attributes=None) -> User:
    atts = dict(attributes or {})

    for a, b in _aliases:
        if a in atts:
            atts[b] = atts[a]
        elif b in atts:
            atts[a] = atts[b]

    if 'displayName' not in atts:
        atts['displayName'] = atts.get('login', '')

    atts['local_uid'] = local_uid
    atts['provider_uid'] = provider.uid

    roles = set(roles) if roles else set()

    return cls(provider, local_uid, roles, atts)
