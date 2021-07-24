"""Authorization and session manager."""

import gws
import gws.types as t
import gws.lib.date
import gws.lib.json2

from . import error
from .stores import sqlite


#

class MethodConfig(gws.Config):
    """Auth method config."""

    secure: bool = True  #: use only with SSL


class Method(gws.Object):
    secure: bool

    def configure(self):
        self.secure = self.var('secure')

    def open_session(self, auth: 'Manager', req: gws.IWebRequest) -> t.Optional['Session']:
        pass

    def close_session(self, auth: 'Manager', sess: 'Session', req: gws.IWebRequest, res: gws.IWebResponse):
        pass

    def login(self, auth: 'Manager', credentials: gws.Data, req: gws.IWebRequest) -> t.Optional['Session']:
        raise error.AccessDenied()

    def logout(self, auth: 'Manager', sess: 'Session', req: gws.IWebRequest) -> 'Session':
        pass


#


class ProviderConfig(gws.Config):
    """Auth provider config."""

    allowedMethods: t.Optional[t.List[str]]  #: allowed authorization methods


class Provider(gws.Object):
    allowed_methods: t.List[str]

    def configure(self):
        self.allowed_methods = self.var('allowedMethods')

    def get_user(self, user_uid: str) -> t.Optional['User']:
        pass

    def authenticate(self, method: 'Method', credentials: gws.Data) -> t.Optional['User']:
        pass

    def user_from_dict(self, d: dict) -> 'User':
        return ValidUser().init_from_data(self, d['user_uid'], d['roles'], d['attributes'])

    def user_to_dict(self, u: 'User') -> dict:
        return {
            'provider_uid': self.uid,
            'user_uid': u.uid,
            'roles': list(u.roles),
            'attributes': u.attributes
        }


# 

class Session:
    def __init__(self, kind: str, user: 'User', method: t.Optional['Method'], uid=None, data=None):
        self.changed = False
        self.data: dict = data or {}
        self.method: t.Optional['Method'] = method
        self.kind: str = kind
        self.uid: str = uid
        self.user: User = user

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val):
        self.data[key] = val
        self.changed = True


#


class Role:
    def __init__(self, name):
        self.name = name

    def can_use(self, obj, parent=None):
        if obj == self:
            return True
        return _can_use([self.name], obj, parent)


#

def make_fid(user):
    return f'{user.provider.uid}::{user.uid}'


def parse_fid(fid):
    s = fid.split('::', 1)
    if len(s) == 2:
        return s
    raise ValueError(f'invalid fid: {fid!r}')


class UserProps(gws.Props):
    displayName: str


class User(gws.IUser):
    attributes: t.Dict[str, t.Any]
    name: str
    provider: 'Provider'
    roles: t.List[str]
    uid: str

    @property
    def display_name(self) -> str:
        return str(self.attributes.get('displayName', ''))

    @property
    def is_guest(self) -> bool:
        return False

    @property
    def fid(self) -> str:
        return make_fid(self)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def init_from_source(self, provider, uid, roles=None, attributes=None) -> 'User':
        atts = dict(attributes or {})

        for a, b in _aliases:
            if a in atts:
                atts[b] = atts[a]
            elif b in atts:
                atts[a] = atts[b]

        if 'displayName' not in atts:
            atts['displayName'] = atts.get('login', '')

        atts['uid'] = uid
        atts['provider_uid'] = provider.uid
        atts['guest'] = self.is_guest

        roles = list(roles) if roles else []
        roles.append(_ROLE_GUEST if self.is_guest else _ROLE_USER)
        roles.append(_ROLE_ALL)

        return self.init_from_data(provider, uid, roles, atts)

    def init_from_data(self, provider, uid, roles, attributes) -> 'User':
        self.attributes = attributes
        self.provider = provider
        self.roles = sorted(set(roles))
        self.uid = uid

        gws.log.debug(f'inited user: prov={provider.uid!r} uid={uid!r} roles={roles!r}')
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
        return gws.Props(
            displayName=self.display_name
        )


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
        gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r}: empty')
        return False

    if _ROLE_ADMIN in roles:
        gws.log.debug(f'PERMS: query: t={_repr(target)} roles={roles!r} found: _ROLE_ADMIN')
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

    gws.log.debug(f'PERMS: query: obj={_repr(target)} roles={roles!r}: not found')
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


#


class Config(gws.Config):
    """Authentication and authorization options"""

    methods: t.Optional[t.List[gws.ext.auth.method.Config]]  #: authorization methods
    providers: t.Optional[t.List[gws.ext.auth.provider.Config]]  #: authorization providers
    sessionLifeTime: gws.Duration = '1200'  #: session life time
    sessionStorage: str = 'sqlite'  #: session storage engine


class Manager(gws.Object, gws.IAuthManager):
    """Authorization manager."""

    session_life_time: int
    guest_user: gws.IUser

    store: sqlite.SessionStore
    providers: t.List[Provider]
    methods: t.List[Method]

    def configure(self):

        self.session_life_time = self.var('sessionLifeTime')

        # if self.var('sessionStorage') == 'sqlite':
        # @TODO other store types
        self.store = sqlite.SessionStore()
        self.store.init()

        p = self.var('providers', default=[])
        self.providers = [t.cast(Provider, self.create_child('gws.ext.auth.provider', c)) for c in p]

        sys = t.cast(Provider, self.create_child('gws.ext.auth.provider', gws.Config(type='system')))
        self.providers.append(sys)
        self.guest_user = sys.get_user('guest')

        # no methods at all, enable the web method
        p = self.var('methods', default=[gws.Config(type='web')])
        self.methods = [t.cast(Method, self.create_child('gws.ext.auth.method', c)) for c in p]

    @property
    def guest_session(self):
        return self.new_session(kind='guest', method=None, user=t.cast('User', self.guest_user))

    # session manager

    def new_session(self, kind, user, method=None, uid=None, data=None):
        return Session(kind, user, method, uid, data)

    def open_session(self, req: gws.IWebRequest) -> Session:
        for m in self.methods:
            sess = m.open_session(self, req)
            if sess:
                return sess
        return self.guest_session

    def close_session(self, sess: Session, req: gws.IWebRequest, res: gws.IWebResponse) -> Session:
        if sess and sess.method:
            sess.method.close_session(self, sess, req, res)
        return self.guest_session

    # stored sessions

    def find_stored_session(self, uid):
        rec = self.store.find(uid)
        if not rec:
            return

        age = gws.lib.date.timestamp() - rec['updated']
        if age > self.session_life_time:
            gws.log.debug(f'sess uid={uid!r} EXPIRED age={age!r}')
            self.store.delete(uid)
            return None

        user = self.unserialize_user(rec['str_user'])
        if not user:
            gws.log.error(f'FAILED to unserialize user from sess={uid!r}')
            return None

        return self.new_session(
            kind=rec['session_kind'],
            uid=rec['uid'],
            method=self.get_method(rec['method_type']),
            user=user,
            data=gws.lib.json2.from_string(rec['str_data'])
        )

    def create_stored_session(self, kind: str, m: 'Method', user: User) -> Session:
        self.store.cleanup(self.session_life_time)

        uid = self.store.create(
            session_kind=kind,
            method_type=m.ext_type,
            provider_uid=user.provider.uid,
            user_uid=user.uid,
            str_user=self.serialize_user(user))

        return self.find_stored_session(uid)

    def save_stored_session(self, sess: Session):
        if sess.changed:
            self.store.update(sess.uid, str_data=gws.lib.json2.to_string(sess.data))
        else:
            self.store.touch(sess.uid)

    def destroy_stored_session(self, sess: Session):
        self.store.delete(sess.uid)

    def delete_stored_sessions(self):
        self.store.delete_all()

    def stored_session_records(self) -> t.List[dict]:
        return self.store.get_all()

    #

    def authenticate(self, method: Method, credentials: gws.Data) -> t.Optional['User']:
        for prov in self.providers:
            if prov.allowed_methods and method.ext_type not in prov.allowed_methods:
                continue
            gws.log.debug(f'trying provider {prov.uid!r}')
            user = prov.authenticate(method, credentials)
            if user:
                return user

    def get_user(self, user_fid: str) -> t.Optional[User]:
        provider_uid, user_uid = parse_fid(user_fid)
        prov = self.get_provider(provider_uid)
        if prov:
            return prov.get_user(user_uid)

    def get_role(self, name: str):
        return Role(name)

    def get_provider(self, uid: str) -> t.Optional['Provider']:
        for prov in self.providers:
            if prov.uid == uid:
                return prov

    def get_method(self, ext_type: str) -> t.Optional['Method']:
        for m in self.methods:
            if m.ext_type == ext_type:
                return m

    def serialize_user(self, user):
        return gws.lib.json2.to_string(t.cast('User', user).provider.user_to_dict(user))

    def unserialize_user(self, s):
        d = gws.lib.json2.from_string(s)
        prov = self.get_provider(d['provider_uid'])
        return prov.user_from_dict(d) if prov else None
