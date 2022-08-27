import gws
import gws.core.tree
import gws.types as t

from . import error

_DELIM = '::'


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
        self.isGuest = gws.ROLE_GUEST in self.roles
        self.displayName = str(self.attributes.get('displayName', ''))
        gws.log.debug(f'inited user: prov={provider.uid!r} local_uid={local_uid!r} roles={roles!r}')

    def props(self, user):
        return gws.Data(displayName=self.displayName)

    def can_use(self, obj, context=None):
        acc = self.access_to(obj)
        if acc is not None:
            return acc == gws.ACCESS_ALLOWED
        obj = context or getattr(obj, 'parent', None)
        while obj:
            acc = self.access_to(obj)
            if acc is not None:
                return acc == gws.ACCESS_ALLOWED
            obj = getattr(obj, 'parent', None)
        return False

    def access_to(self, obj):
        access = getattr(obj, 'access', None)
        if access:
            for a, r in access:
                if r in self.roles:
                    return a


class Guest(User):
    def __init__(self, provider, local_uid, roles, attributes):
        super().__init__(provider, local_uid, roles, attributes)
        self.roles.add(gws.ROLE_GUEST)
        self.isGuest = True


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
