import gws
import gws.lib.jsonx
import gws.types as t


class Props(gws.Props):
    displayName: str


_ALIASES = [
    # https://tools.ietf.org/html/rfc4519
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


class User(gws.Object, gws.IUser):
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
        if gws.ROLE_ADMIN in self.roles:
            return True

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

    def require(self, uid, classref=None, access=gws.Access.use):
        obj = self.provider.root.get(uid, classref)
        if not obj:
            raise gws.Error(f'required object {classref} {uid} not found')
        if not self.can(access, obj):
            raise gws.Error(f'required object {classref} {uid} denied')
        return obj

    def acquire(self, uid, classref=None, access=gws.Access.use):
        obj = self.provider.root.get(uid, classref)
        if obj and self.can(access, obj):
            return obj


class Guest(User):
    isGuest = True


class System(User):
    def can(self, access, obj, *context):
        return True


class Nobody(User):
    def can(self, access, obj, *context):
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
        providerUid=usr.provider.uid,
        roles=list(usr.roles),
        uid=usr.uid,
    )


def from_dict(cls, provider: gws.IAuthProvider, d: dict) -> User:
    usr = cls(provider)

    usr.attributes = d.get('attributes', {})
    usr.displayName = d.get('displayName', '')
    usr.localUid = d.get('localUid', '')
    usr.loginName = d.get('loginName', '')
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

    atts = dict(kwargs.get('attributes', {}))

    for a, b in _ALIASES:
        if a in atts:
            atts[b] = atts[a]
        elif b in atts:
            atts[a] = atts[b]

    usr.attributes = atts

    return usr
