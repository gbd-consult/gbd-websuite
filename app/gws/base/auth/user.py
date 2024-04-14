import gws
import gws.lib.jsonx

import gws.types as t


class Props(gws.Props):
    displayName: str
    attributes: dict


class User(gws.User):
    isGuest = False

    def __init__(self, provider, roles):
        self.attributes = {}
        self.authToken = ''
        self.displayName = ''
        self.localUid = ''
        self.loginName = ''
        self.uid = ''

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

    def can_edit(self, obj, *context):
        return (
                self.can(gws.Access.write, obj, *context)
                or self.can(gws.Access.create, obj, *context)
                or self.can(gws.Access.delete, obj, *context)
        )

    def can_delete(self, obj, *context):
        return self.can(gws.Access.delete, obj, *context)

    def can(self, access, obj, *context):
        ci = 0
        clen = len(context)

        while obj:
            bit = self.acl_bit(access, obj)
            if bit is not None:
                return bit == gws.c.ALLOW
            obj = context[ci] if ci < clen else getattr(obj, 'parent', None)
            ci += 1

        return False

    def acl_bit(self, access, obj):
        if obj is self and access == gws.Access.read:
            return gws.c.ALLOW
        acl = obj.permissions.get(access)
        if acl:
            for bit, role in acl:
                if role in self.roles:
                    return bit

    def require(self, uid=None, classref=None, access=None):
        access = access or gws.Access.read
        obj = self.provider.root.get(uid, classref)
        if not obj:
            raise gws.NotFoundError(f'required object {classref} {uid} not found')
        if not self.can(access, obj):
            raise gws.ForbiddenError(f'required object {classref} {uid} forbidden')
        return obj

    def acquire(self, uid=None, classref=None, access=None):
        access = access or gws.Access.read
        obj = self.provider.root.get(uid, classref)
        if obj and self.can(access, obj):
            return obj

    def require_project(self, uid=None):
        return t.cast(gws.Project, self.require(uid, gws.ext.object.project))

    def require_layer(self, uid=None):
        return t.cast(gws.Layer, self.require(uid, gws.ext.object.layer))


class GuestUser(User):
    isGuest = True


class SystemUser(User):
    def acl_bit(self, access, obj):
        return gws.c.ALLOW


class NobodyUser(User):
    def acl_bit(self, access, obj):
        return gws.c.DENY


class AuthorizedUser(User):
    pass


class AdminUser(User):
    def acl_bit(self, access, obj):
        return gws.c.ALLOW


##


##

def to_dict(usr) -> dict:
    return dict(
        attributes=usr.attributes,
        authToken=usr.authToken,
        displayName=usr.displayName,
        localUid=usr.localUid,
        loginName=usr.loginName,
        providerUid=usr.provider.uid,
        roles=list(usr.roles),
        uid=usr.uid,
    )


def from_dict(provider: gws.AuthProvider, d: dict) -> gws.User:
    roles = set(d.get('roles', []))

    if gws.c.ROLE_GUEST in roles:
        return provider.root.app.authMgr.guestUser
    if gws.c.ROLE_ADMIN in roles:
        usr = AdminUser(provider, roles)
    else:
        usr = AuthorizedUser(provider, roles)

    usr.attributes = d.get('attributes', {})
    usr.authToken = d.get('authToken', '')
    usr.displayName = d.get('displayName', '')
    usr.localUid = d.get('localUid', '')
    usr.loginName = d.get('loginName', '')
    usr.uid = gws.u.join_uid(provider.uid, usr.localUid)

    return usr


def init(provider: gws.AuthProvider, **kwargs) -> gws.User:
    roles = set(kwargs.get('roles', []))
    roles.add(gws.c.ROLE_ALL)

    if gws.c.ROLE_GUEST in roles:
        return provider.root.app.authMgr.guestUser
    if gws.c.ROLE_ADMIN in roles:
        usr = AdminUser(provider, roles)
    else:
        roles.add(gws.c.ROLE_USER)
        usr = AuthorizedUser(provider, roles)

    usr.attributes = _process_aliases(dict(kwargs.get('attributes', {})))
    usr.authToken = kwargs.get('authToken') or ''
    usr.displayName = kwargs.get('displayName') or kwargs.get('loginName') or ''
    usr.localUid = kwargs.get('localUid') or kwargs.get('loginName') or ''
    usr.loginName = kwargs.get('loginName') or ''
    usr.uid = gws.u.join_uid(provider.uid, usr.localUid)

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
