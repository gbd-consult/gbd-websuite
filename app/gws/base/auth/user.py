import gws
import gws.lib.jsonx


class Props(gws.Props):
    displayName: str
    attributes: dict


class User(gws.Object, gws.IUser):
    isGuest = False

    def __init__(self, provider, roles):
        self.provider = provider
        self.roles = roles

    def props(self, user):
        return Props(displayName=self.displayName, attributes=self.attributes)

    def can_use(self, obj, *context):
        return self.can(gws.Access.read, obj, *context)

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
        if obj is self and access == gws.Access.read:
            return gws.ALLOW
        acl = obj.permissions.get(access)
        if acl:
            for bit, role in acl:
                if role in self.roles:
                    return bit

    def require(self, uid, classref=None, access=gws.Access.read):
        obj = self.provider.root.get(uid, classref)
        if not obj:
            raise gws.NotFoundError(f'required object {classref} {uid} not found')
        if not self.can(access, obj):
            raise gws.ForbiddenError(f'required object {classref} {uid} forbidden')
        return obj

    def acquire(self, uid, classref=None, access=gws.Access.read):
        obj = self.provider.root.get(uid, classref)
        if obj and self.can(access, obj):
            return obj


class GuestUser(User):
    isGuest = True


class SystemUser(User):
    def acl_bit(self, access, obj):
        return gws.ALLOW


class NobodyUser(User):
    def acl_bit(self, access, obj):
        return gws.DENY


class AuthorizedUser(User):
    pass


class AdminUser(User):
    def acl_bit(self, access, obj):
        return gws.ALLOW


##


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


def from_dict(provider: gws.IAuthProvider, d: dict) -> gws.IUser:
    roles = set(d.get('roles', []))

    if gws.ROLE_GUEST in roles:
        return provider.authMgr.guestUser
    if gws.ROLE_ADMIN in roles:
        usr = AdminUser(provider, roles)
    else:
        usr = AuthorizedUser(provider, roles)

    usr.attributes = d.get('attributes', {})
    usr.displayName = d.get('displayName', '')
    usr.localUid = d.get('localUid', '')
    usr.loginName = d.get('loginName', '')
    usr.uid = gws.join_uid(provider.uid, usr.localUid)

    return usr


def init(provider: gws.IAuthProvider, **kwargs) -> gws.IUser:
    roles = set(kwargs.get('roles', []))
    roles.add(gws.ROLE_ALL)

    if gws.ROLE_GUEST in roles:
        return provider.authMgr.guestUser
    if gws.ROLE_ADMIN in roles:
        usr = AdminUser(provider, roles)
    else:
        roles.add(gws.ROLE_USER)
        usr = AuthorizedUser(provider, roles)

    usr.attributes = _process_aliases(dict(kwargs.get('attributes', {})))
    usr.displayName = kwargs.get('displayName') or kwargs.get('loginName') or ''
    usr.localUid = kwargs.get('localUid') or kwargs.get('loginName') or ''
    usr.loginName = kwargs.get('loginName')
    usr.uid = gws.join_uid(provider.uid, usr.localUid)

    if not usr.localUid:
        raise gws.Error(f'missing local uid for user', kwargs)

    return usr


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


def _process_aliases(atts):
    for a, b in _ALIASES:
        if a in atts:
            atts[b] = atts[a]
        elif b in atts:
            atts[a] = atts[b]
    return atts
