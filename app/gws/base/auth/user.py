import gws
import gws.lib.jsonx
import gws.types as t

from . import error

_DELIM = '::'


def make_uid(user):
    return f'{user.provider.uid}{_DELIM}{user.localUid}'


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

_GUEST_ROLES = set([gws.ROLE_GUEST, gws.ROLE_ALL])


class User(gws.IUser):
    isGuest = False

    def __init__(self, provider):
        self.provider = provider

    def props(self, user):
        return gws.Data(displayName=self.displayName, attributes=self.attributes)

    def can_use(self, obj, *context):
        if obj is self:
            return True

        ci = 0
        clen = len(context)

        while obj:
            acc = self.access_to(obj)
            if acc is not None:
                return acc == gws.ALLOW
            obj = context[ci] if ci < clen else getattr(obj, 'parent', None)
            ci += 1

        return False

    def access_to(self, obj):
        roles = _GUEST_ROLES if self.pendingMfa else self.roles
        if gws.ROLE_ADMIN in roles:
            return gws.ALLOW
        access = getattr(obj, 'access', None)
        if access:
            for bit, role in access:
                if role in roles:
                    return bit


class Guest(User):
    isGuest = True


class System(User):
    def can_use(self, obj, *context):
        return True


class Nobody(User):
    def can_use(self, obj, *context):
        return False


class AuthorizedUser(User):
    pass


##

def to_dict(usr) -> dict:
    return dict(
        attributes=usr.attributes,
        displayName=usr.displayName,
        localUid=usr.localUid,
        loginName=usr.loginName,
        pendingMfa=gws.to_dict(usr.pendingMfa),
        providerUid=usr.provider.uid,
        roles=list(usr.roles),
        uid=usr.uid,
    )


def from_dict(cls, provider: gws.IAuthProvider, d: dict) -> User:
    usr = cls(provider)
    usr.attributes = d.get('attributes')
    usr.displayName = d.get('displayName')
    usr.localUid = d.get('localUid')
    usr.loginName = d.get('loginName')
    usr.pendingMfa = gws.Data(d.get('pendingMfa')) if d.get('pendingMfa') else None
    usr.roles = set(d.get('roles'))
    usr.uid = d.get('uid')
    return usr


def from_args(cls, provider: gws.IAuthProvider, **kwargs) -> User:
    usr = cls(provider)

    roles = set(kwargs.get('roles', []))
    roles.add(gws.ROLE_ALL)
    usr.roles = roles

    usr.localUid = kwargs.get('localUid') or kwargs.get('loginName')
    usr.loginName = kwargs.get('loginName') or kwargs.get('localUid')
    usr.displayName = kwargs.get('displayName') or usr.loginName

    usr.uid = make_uid(usr)

    p = kwargs.get('pendingMfa')
    usr.pendingMfa = gws.AuthPendingMfa(p) if p else None

    atts = dict(kwargs.get('attributes', {}))

    for a, b in _aliases:
        if a in atts:
            atts[b] = atts[a]
        elif b in atts:
            atts[a] = atts[b]

    usr.attributes = atts

    gws.log.debug(f'inited user: prov={provider.uid!r} localUid={usr.localUid!r}')

    return usr
