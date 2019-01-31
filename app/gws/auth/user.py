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


def _can_use(who, obj, roles):
    if ADMIN_ROLE_NAME in roles:
        return True

    if not obj:
        return False

    if obj == who:
        return True

    cur = obj
    while cur:
        access = gws.get(cur, 'access')
        
        if access:
            for a in access:
                for role in roles:
                    if role == a.role:
                        gws.log.debug(
                            f'PERMS: query: obj={obj!r} roles={roles!r} found: {a.role}:{a.type} in {cur!r}')
                        return a.type == 'allow'
        
        cur = gws.get(cur, 'parent')

    gws.log.debug('PERMS: NOT found')
    return False


class Role:
    def __init__(self, name):
        self.name = name

    def can_use(self, obj):
        return _can_use(self, obj, [self.name])


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

    def can_use(self, obj):
        return _can_use(self, obj, self.roles)

    @property
    def props(self):
        return None


class Guest(User):
    @property
    def is_guest(self):
        return True


class System(User):
    def can_use(self, obj):
        gws.log.debug(f'PERMS: sys_allow : obj={obj!r}')
        return True


class Nobody(User):
    def can_use(self, obj):
        gws.log.debug(f'PERMS: sys_deny : obj={obj!r}')
        return False


class Props(t.Data):
    displayName: str


class ValidUser(User):
    @property
    def props(self):
        return {
            'displayName': self.display_name
        }
