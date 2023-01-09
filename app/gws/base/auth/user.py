import gws
import gws.lib.jsonx
import gws.types as t


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
        return self.can(gws.Access.use, obj, *context)

    def can_read(self, obj, *context):
        return self.can(gws.Access.read, obj, *context)

    def can_write(self, obj, *context):
        return self.can(gws.Access.write, obj, *context)

    def can_create(self, obj, *context):
        return self.can(gws.Access.create, obj, *context)

    def can_delete(self, obj, *context):
        return self.can(gws.Access.delete, obj, *context)

    def can(self, access, obj, *context):
        ci = 0
        clen = len(context)

        while obj:
            bit = self.acl_bit(access, obj)
            if bit is not None:
                return bit == gws.ALLOW
            obj = context[ci] if ci < clen else getattr(obj, 'parent', None)
            ci += 1

        return False

    def acl_bit(self, access, obj):
        perms = getattr(obj, 'permissions', None)
        acl = perms.get(access) if perms else None
        if acl:
            for bit, role in acl:
                if role in self.roles:
                    return bit


class Guest(User):
    isGuest = True


class System(User):
    def acl_bit(self, access, obj):
        return gws.ALLOW


class Nobody(User):
    def acl_bit(self, access, obj):
        return gws.DENY


class AuthorizedUser(User):
    pass


##

def to_dict(usr) -> dict:
    return dict(
        attributes=usr.attributes,
        displayName=usr.displayName,
        localUid=usr.localUid,
        loginName=usr.loginName,
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
    usr.roles = set(d.get('roles'))
    usr.uid = gws.join_uid(provider.uid, usr.localUid)
    return usr


def from_args(cls, provider: gws.IAuthProvider, **kwargs) -> User:
    usr = cls(provider)

    roles = set(kwargs.get('roles', []))
    roles.add(gws.ROLE_ALL)
    usr.roles = roles

    usr.localUid = kwargs.get('localUid') or kwargs.get('loginName')
    usr.loginName = kwargs.get('loginName') or kwargs.get('localUid')
    usr.displayName = kwargs.get('displayName') or usr.loginName

    usr.uid = gws.join_uid(provider.uid, usr.localUid)

    p = kwargs.get('pendingMfa')
    usr.pendingMfa = gws.AuthPendingMfa(p) if p else None

    atts = dict(kwargs.get('attributes', {}))

    for a, b in _aliases:
        if a in atts:
            atts[b] = atts[a]
        elif b in atts:
            atts[a] = atts[b]

    usr.attributes = atts

    return usr
