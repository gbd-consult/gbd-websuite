import gws
import gws.config
import gws.tools.json2 as json2
import gws.types as t

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

ADMIN_ROLE_NAME = 'admin'


def _check_access(obj, cur, roles):
    access = gws.get(cur, 'access')

    if not access:
        return

    for a in access:
        if a.role in roles:
            gws.log.debug(f'PERMS: query: obj={_repr(obj)} roles={roles!r} found: {a.role}:{a.type} in {_repr(cur)}')
            return a.type == 'allow'


def _can_use(who, obj, roles, parent):
    if not obj:
        gws.log.debug(f'PERMS: query: obj={_repr(obj)} roles={roles!r}: empty')
        return False

    if ADMIN_ROLE_NAME in roles:
        gws.log.debug(f'PERMS: query: obj={_repr(obj)} roles={roles!r} found: ADMIN_ROLE_NAME')
        return True

    if obj == who:
        return True

    c = _check_access(obj, obj, roles)
    if c is not None:
        return c

    cur = parent or gws.get(obj, 'parent')

    while cur:
        c = _check_access(obj, cur, roles)
        if c is not None:
            return c
        cur = gws.get(cur, 'parent')

    gws.log.debug(f'PERMS: query: obj={_repr(obj)} roles={roles!r}: not found')
    return False


class Role:
    def __init__(self, name):
        self.name = name

    def can_use(self, obj, parent=None):
        return _can_use(self, obj, [self.name], parent)


class User(gws.PublicObject, t.AuthUserInterface):
    attributes = {}
    provider = None
    roles = None
    uid = ''

    @property
    def display_name(self):
        return self.attributes.get('displayName')

    @property
    def is_guest(self):
        return False

    @property
    def full_uid(self):
        return json2.to_string([self.provider.uid, self.uid])

    def init_from_source(self, provider, uid, roles=None, attributes=None):
        attributes = dict(attributes) if attributes else {}

        for a, b in _aliases:
            if a in attributes:
                attributes[b] = attributes[a]
            elif b in attributes:
                attributes[a] = attributes[b]

        if 'displayName' not in attributes:
            attributes['displayName'] = attributes.get('login', '')

        attributes['uid'] = uid
        attributes['provider'] = provider.uid
        attributes['guest'] = self.is_guest

        roles = list(roles) if roles else []
        roles.append('guest' if self.is_guest else 'user')
        roles.append('all')

        return self.init_from_cache(provider, uid, roles, attributes)

    def init_from_cache(self, provider, uid, roles, attributes):
        self.attributes = attributes
        self.provider = provider
        self.roles = set(roles)
        self.uid = uid

        gws.log.info(f'inited user: prov={provider.uid!r} uid={uid!r} roles={roles!r}')
        return self

    def attribute(self, key, default=''):
        return self.attributes.get(key, default)

    def can_use(self, obj, parent=None):
        return _can_use(self, obj, self.roles, parent)

    @property
    def props(self):
        return None


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


class Props(t.Data):
    displayName: str


class ValidUser(User):
    @property
    def props(self):
        return {
            'displayName': self.display_name
        }


def _repr(obj):
    if not obj:
        return repr(obj)
    return repr(gws.get(obj, 'uid') or obj)
