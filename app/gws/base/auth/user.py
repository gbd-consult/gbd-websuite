from typing import Optional, cast

import gws
import gws.lib.jsonx


class Props(gws.Props):
    displayName: str
    attributes: dict


_FIELDS = {
    'authToken',
    'displayName',
    'email',
    'localUid',
    'loginName',
    'mfaSecret',
    'mfaUid',
}


class User(gws.User):
    isGuest = False

    def __init__(self, provider, roles):
        super().__init__()

        self.authProvider = provider

        self.attributes = {}
        self.data = {}
        self.roles = roles
        self.uid = ''

        for f in _FIELDS:
            setattr(self, f, '')

    def props(self, user):
        return Props(displayName=self.displayName, attributes=self.attributes)

    def has_role(self, role):
        return role in self.roles

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
        obj = self.authProvider.root.get(uid, classref)
        if not obj:
            raise gws.NotFoundError(f'required object {classref} {uid} not found')
        if not self.can(access, obj):
            raise gws.ForbiddenError(f'required object {classref} {uid} forbidden')
        return obj

    def acquire(self, uid=None, classref=None, access=None):
        access = access or gws.Access.read
        obj = self.authProvider.root.get(uid, classref)
        if obj and self.can(access, obj):
            return obj

    def require_project(self, uid=None):
        return cast(gws.Project, self.require(uid, gws.ext.object.project))

    def require_layer(self, uid=None):
        return cast(gws.Layer, self.require(uid, gws.ext.object.layer))


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
    d = {}

    d['attributes'] = usr.attributes or {}
    d['data'] = usr.data or {}
    d['roles'] = list(usr.roles)
    d['uid'] = usr.uid

    for f in _FIELDS:
        d[f] = getattr(usr, f, '')

    return d


def from_dict(provider: gws.AuthProvider, d: dict) -> gws.User:
    roles = set(d.get('roles', []))

    if gws.c.ROLE_GUEST in roles:
        return provider.root.app.authMgr.guestUser

    if gws.c.ROLE_ADMIN in roles:
        usr = AdminUser(provider, roles)
    else:
        usr = AuthorizedUser(provider, roles)

    for f in _FIELDS:
        setattr(usr, f, d.get(f, ''))

    usr.attributes = d.get('attributes', {})
    usr.data = d.get('data', {})
    usr.roles = roles
    usr.uid = gws.u.join_uid(provider.uid, usr.localUid)

    return usr


def from_record(provider: gws.AuthProvider, user_rec: dict) -> gws.User:
    """Create a User from a raw record as returned from a provider.

    A provider can return an arbitrary dict of values. Entries whose keys are
    in the `_FIELDS` list (case-insensitively), are copied to the newly
    created `User` object.

    Entries ``roles`` and ``attributes`` are copied as well,
    other entries are stored in the user's ``data`` dict.
    """

    data = dict(user_rec)

    roles = set(gws.u.to_list(data.pop('roles', [])))
    roles.add(gws.c.ROLE_ALL)

    if gws.c.ROLE_GUEST in roles:
        return provider.root.app.authMgr.guestUser

    if gws.c.ROLE_ADMIN in roles:
        usr = AdminUser(provider, roles)
    else:
        roles.add(gws.c.ROLE_USER)
        usr = AuthorizedUser(provider, roles)

    for f in _FIELDS:
        if f in data:
            setattr(usr, f, data.pop(f))
            continue
        if f.lower() in data:
            setattr(usr, f, data.pop(f.lower()))
            continue

    usr.attributes = data.pop('attributes', {})
    usr.data = _process_aliases(data)

    if not usr.loginName and 'login' in usr.data:
        usr.loginName = usr.data['login']

    if not usr.email and 'email' in usr.data:
        usr.email = usr.data['email']

    usr.localUid = usr.localUid or usr.loginName
    if not usr.localUid:
        raise gws.Error(f'missing local uid for user')

    usr.displayName = usr.displayName or usr.loginName

    usr.uid = gws.u.join_uid(provider.uid, usr.localUid)

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
    ('mail', 'email'),
]


def _process_aliases(r):
    for a, b in _ALIASES:
        if a in r:
            r[b] = r[a]
        elif b in r:
            r[a] = r[b]
    return r
