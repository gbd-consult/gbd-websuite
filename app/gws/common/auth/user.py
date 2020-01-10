import gws
import gws.config
import gws.tools.json2 as json2
import gws.types as t


#:export
class UserProps(t.Data):
    displayName: str


#:export IRole
class Role(t.IRole):
    def __init__(self, name):
        self.name = name

    def can_use(self, obj, parent=None):
        return _can_use(self, obj, [self.name], parent)


#:export IUser
class User(t.IUser):
    def __init__(self):
        self.attributes = {}
        self.provider: t.IAuthProvider = None
        self.roles: t.List[str] = []
        self.uid = ''

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
    def full_uid(self) -> str:
        return json2.to_string([self.provider.uid, self.uid])

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def init_from_source(self, provider, uid, roles=None, attributes=None) -> t.IUser:
        attributes = dict(attributes) if attributes else {}

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

        return self.init_from_props(provider, uid, roles, attributes)

    def init_from_props(self, provider, uid, roles, attributes) -> t.IUser:
        self.attributes = attributes
        self.provider = provider
        self.roles = set(roles)
        self.uid = uid

        gws.log.info(f'inited user: prov={provider.uid!r} uid={uid!r} roles={roles!r}')
        return self

    def attribute(self, key: str, default: str = '') -> str:
        return self.attributes.get(key, default)

    def can_use(self, obj, parent=None) -> bool:
        return _can_use(self, obj, self.roles, parent)


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


def _can_use(who, target, roles, parent):
    if not target:
        gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r}: empty')
        return False

    if _ROLE_ADMIN in roles:
        gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: _ROLE_ADMIN')
        return True

    if target == who:
        return True

    c = _check_access(target, target, roles)
    if c is not None:
        return c

    current = parent or gws.get(target, 'parent')

    while current:
        c = _check_access(target, current, roles)
        if c is not None:
            return c
        current = gws.get(current, 'parent')

    gws.log.debug(f'PERMS: query: obj={_repr(target)} roles={roles!r}: not found')
    return False


def _check_access(target, current, roles):
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
