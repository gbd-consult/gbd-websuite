import gws
import gws.config
import gws.types as t


#:export
class UserProps(t.Data):
    displayName: str


#:export IRole
class Role(t.IRole):
    def __init__(self, name):
        self.name = name

    def can_use(self, obj, parent=None):
        if obj == self:
            return True
        return _can_use([self.name], obj, parent)


def make_fid(user):
    return f'{user.provider.uid}::{user.uid}'


def parse_fid(fid):
    s = fid.split('::', 1)
    if len(s) == 2:
        return s
    raise ValueError(f'invalid fid: {fid!r}')


#:export IUser
class User(t.IUser):
    attributes = {}
    provider: t.IAuthProvider
    roles: t.List[str] = []
    uid = ''

    @property
    def props(self) -> t.UserProps:
        return t.UserProps()

    @property
    def display_name(self) -> str:
        return self.attributes.get('displayName')

    @property
    def is_guest(self) -> bool:
        return False

    @property
    def fid(self) -> str:
        return make_fid(self)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def init_from_source(self, provider, uid, roles=None, attributes=None) -> t.IUser:
        attributes = dict(attributes or {})

        for a, b in _aliases:
            if a in attributes:
                attributes[b] = attributes[a]
            elif b in attributes:
                attributes[a] = attributes[b]

        if 'displayName' not in attributes:
            attributes['displayName'] = attributes.get('login', '')

        attributes['uid'] = uid
        attributes['provider_uid'] = provider.uid
        attributes['guest'] = self.is_guest

        roles = list(roles) if roles else []
        roles.append(_ROLE_GUEST if self.is_guest else _ROLE_USER)
        roles.append(_ROLE_ALL)

        return self.init_from_data(provider, uid, roles, attributes)

    def init_from_data(self, provider, uid, roles, attributes) -> t.IUser:
        self.attributes = attributes
        self.provider = provider
        self.roles = sorted(set(roles))
        self.uid = uid

        # gws.log.info(f'inited user: prov={provider.uid!r} uid={uid!r} roles={roles!r}')
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


class ValidUser(User):
    @property
    def props(self):
        return UserProps({
            'displayName': self.display_name
        })


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

_ROLE_ADMIN = 'admin'
_ROLE_USER = 'user'
_ROLE_GUEST = 'guest'
_ROLE_ALL = 'all'


def _can_use(roles, target, parent):
    if not target:
        # gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r}: empty')
        return False

    if _ROLE_ADMIN in roles:
        # gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: _ROLE_ADMIN')
        return True

    c = _check_access(roles, target)
    if c is not None:
        # gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: {c}')
        return c

    current = parent or gws.get(target, 'parent')

    while current:
        c = _check_access(roles, current)
        if c is not None:
            # gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: {c} in {_repr(current)}')
            return c
        current = gws.get(current, 'parent')

    # gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r}: not found')
    return False


def _check_access(roles, current):
    access = gws.get(current, 'access')

    if not access:
        return

    for a in access:
        if a.role in roles:
            return a.type == 'allow'


def _repr(obj):
    if not obj:
        return repr(obj)
    return repr(gws.get(obj, 'uid') or obj.__class__.__name__)
