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
        gws.log.debug(f'PERMS: sys_allow : obj={_repr(obj)}')
        return True


class Nobody(User):
    def can_use(self, obj, parent=None):
        gws.log.debug(f'PERMS: sys_deny : obj={_repr(obj)}')
        return False


class AuthorizedUser(User):
    @property
    def props(self):
        return gws.Props(
            displayName=self.display_name
        )


def _can_use(roles, target, parent):
    if not target:
        gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r}: empty')
        return False

    if ROLE_ADMIN in roles:
        gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: ROLE_ADMIN')
        return True

    c = _check_access(roles, target, target)
    if c is not None:
        return c

    current = parent or gws.get(target, 'parent')

    while current:
        c = _check_access(roles, target, current)
        if c is not None:
            return c
        current = gws.get(current, 'parent')

    gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r}: not found')
    return False


def _check_access(roles, target, current):
    access = gws.get(current, 'access')

    if not access:
        return

    for a in access:
        if a.role in roles:
            gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: {a.role}:{a.type} in {_repr(current)}')
            return a.type == 'allow'


def _repr(obj):
    if not obj:
        return repr(obj)
    return repr(gws.get(obj, 'uid') or obj)
