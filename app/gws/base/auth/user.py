import gws
import gws.types as t

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLE_GUEST = 'guest'
ROLE_ALL = 'all'


class Role(gws.IRole):
    def __init__(self, name):
        self.name = name

    def can_use(self, obj, parent=None):
        if obj == self:
            return True
        return _can_use([self.name], obj, parent)


##

_DELIM = '___'


def make_uid(user):
    return f'{user.provider.uid}{_DELIM}{user.local_uid}'


def parse_uid(user_uid):
    s = user_uid.split(_DELIM, 1)
    if len(s) == 2:
        return s
    raise gws.Error(f'invalid user uid: {user_uid!r}')


##

class UserProps(gws.Props):
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
    attributes: t.Dict[str, t.Any]
    name: str
    provider: gws.IAuthProvider
    roles: t.List[str]
    local_uid: str

    @property
    def display_name(self) -> str:
        return str(self.attributes.get('displayName', ''))

    @property
    def is_guest(self) -> bool:
        return False

    @property
    def uid(self) -> str:
        return make_uid(self)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def init_from_source(self, provider, local_uid, roles=None, attributes=None) -> gws.IUser:
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
        atts['guest'] = self.is_guest

        roles = list(roles) if roles else []
        roles.append(ROLE_GUEST if self.is_guest else ROLE_USER)
        roles.append(ROLE_ALL)

        return self.init_from_data(provider, local_uid, roles, atts)

    def init_from_data(self, provider, local_uid, roles, attributes) -> 'User':
        self.attributes = attributes
        self.provider = provider
        self.roles = sorted(set(roles))
        self.local_uid = local_uid

        gws.log.debug(f'inited user: prov={provider.uid!r} local_uid={local_uid!r} roles={roles!r}')
        return self

    def attribute(self, key: str, default: str = '') -> str:
        return self.attributes.get(key, default)

    def can_use(self, obj, parent=None) -> bool:
        if obj == self:
            return True
        return _can_use(self.roles, obj, parent)


class Guest(User):
    @property
    def is_guest(self):
        return True


class System(User):
    def can_use(self, obj, parent=None):
        return True


class Nobody(User):
    def can_use(self, obj, parent=None):
        return False


class AuthorizedUser(User):
    @property
    def props(self):
        return gws.Props(
            displayName=self.display_name
        )


def _can_use(roles, obj, parent):
    def _repr(x):
        r = repr(gws.get(x, 'uid') or x)
        if len(r) > 200:
            r = r[:200] + '...'
        return r

    if not obj:
        gws.log.debug(f'PERMS:False: o={_repr(obj)} r={roles!r}: EMPTY')
        return False

    if ROLE_ADMIN in roles:
        gws.log.debug(f'PERMS:True o={_repr(obj)} r={roles!r}: ADMIN')
        return True

    c = _check_access(roles, obj)
    if c is not None:
        gws.log.debug(f'PERMS:{c} o={_repr(obj)} r={roles!r}: FOUND')
        return c

    curr = parent or gws.get(obj, 'parent')
    while curr:
        c = _check_access(roles, curr)
        if c is not None:
            gws.log.debug(f'PERMS:{c} o={_repr(obj)} r={roles!r}: IN={_repr(curr)}')
            return c
        curr = gws.get(curr, 'parent')

    gws.log.debug(f'PERMS:False: o={_repr(obj)} r={roles!r}: NOT_FOUND')
    return False


def _check_access(roles, obj):
    access = gws.get(obj, 'access')
    if access:
        for a in access:
            if a.role in roles:
                return a.type == 'allow'
