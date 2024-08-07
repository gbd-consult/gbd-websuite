"""LDAP authorization provider.

Accepts an LDAP URL in the following form::

    ldap://host:port/baseDN?searchAttribute

which is a subset of the rfc2255 schema.

Optionally, a bind dn and a password can be provided. This dn must have search permissions for the directory.

The authorization workflow with the (login, password) credentials is as follows:

- connect to the LDAP server, using the bind dn if provided
- search for the dn matching ``searchAttribute = credentials.login``
- attempt to login with that dn and ``credentials.password``
- iterate the ``users`` configs to determine roles for the user


References:
    https://datatracker.ietf.org/doc/html/rfc2255

"""

from typing import Optional

import contextlib

import ldap
import ldap.filter

import gws
import gws.base.auth
import gws.lib.net


gws.ext.new.authProvider('ldap')


class UserSpec(gws.Data):
    """Map LDAP filters to authorization roles"""

    roles: list[str]
    """GWS role names"""
    matches: Optional[str]
    """LDAP filter the account has to match"""
    memberOf: Optional[str]
    """LDAP group the account has to be a member of"""


class Config(gws.base.auth.provider.Config):
    """LDAP authorization provider"""

    activeDirectory: bool = True
    """True if the LDAP server is ActiveDirectory."""
    bindDN: Optional[str]
    """Bind DN."""
    bindPassword: Optional[str]
    """Bind password."""
    displayNameFormat: Optional[gws.FormatStr]
    """Format for user's display name."""
    users: list[UserSpec]
    """Map LDAP filters to gws roles."""
    timeout: gws.Duration = 30
    """LDAP server timeout."""
    url: str
    """LDAP server url."""


class Object(gws.base.auth.provider.Object):
    serverUrl: str
    baseDN: str
    loginAttribute: str
    timeout: int

    def configure(self):
        p = gws.lib.net.parse_url(self.cfg('url'))

        self.serverUrl = 'ldap://' + p.netloc
        self.baseDN = p.path.strip('/')
        self.loginAttribute = p.query

        self.timeout = self.cfg('timeout', default=30)

        try:
            with self._connection():
                gws.log.debug(f'LDAP connection {self.uid!r} ok')
        except Exception as e:
            raise gws.Error(f'LDAP connection error: {e.__class__.__name__}', *e.args)

    def authenticate(self, method, credentials):
        username = credentials.get('username')
        password = credentials.get('password')
        if not username or not password:
            return

        with self._connection() as conn:
            users = self._find(conn, _make_filter({self.loginAttribute: username}))

            if len(users) == 0:
                return
            if len(users) > 1:
                raise gws.ForbiddenError(f'multiple entries for {username!r}')

            rec = users[0]

            # check for AD disabled accounts
            uac = str(rec.get('userAccountControl', ''))
            if uac and uac.isdigit():
                if int(uac) & _MS_ACCOUNTDISABLE:
                    raise gws.ForbiddenError('ACCOUNTDISABLE flag set')

            try:
                conn.simple_bind_s(rec['dn'], password)
            except ldap.INVALID_CREDENTIALS:
                raise gws.ForbiddenError(f'wrong password for {username!r}')
            except ldap.LDAPError as exc:
                gws.log.exception()
                raise gws.ForbiddenError(f'LDAP error {exc!r}')

            return self._make_user(conn, rec)

    def get_user(self, local_uid):
        with self._connection() as conn:
            users = self._find(conn, _make_filter({self.loginAttribute: local_uid}))
            if len(users) == 1:
                return self._make_user(conn, users[0])

    ##

    def _make_user(self, conn, rec):
        user_rec = dict(rec)
        user_rec['roles'] = self._roles_for_user(conn, rec)

        if not user_rec.get('displayName') and self.cfg('displayNameFormat'):
            user_rec['displayName'] = gws.u.format_map(self.cfg('displayNameFormat'), rec)

        login = user_rec.pop(self.loginAttribute, '')
        user_rec['localUid'] = user_rec['loginName'] = login

        return gws.base.auth.user.from_record(self, user_rec)

    def _roles_for_user(self, conn, rec):
        user_dn = rec['dn']
        roles = set()

        for u in self.cfg('users'):

            if u.get('matches'):
                for dct in self._find(conn, u.matches):
                    if dct['dn'] == user_dn:
                        roles.update(u.roles)

            elif u.get('memberOf'):
                for dct in self._find(conn, u.memberOf):
                    if _is_member_of(dct, user_dn):
                        roles.update(u.roles)

        return sorted(roles)

    def _find(self, conn, flt):
        res = conn.search_s(self.baseDN, ldap.SCOPE_SUBTREE, flt)
        dcts = []

        for dn, data in res:
            if dn:
                d = _as_dict(data)
                d['dn'] = dn
                dcts.append(d)

        return dcts

    @contextlib.contextmanager
    def _connection(self):
        conn = ldap.initialize(self.serverUrl)
        conn.set_option(ldap.OPT_NETWORK_TIMEOUT, self.timeout)

        if self.cfg('activeDirectory'):
            # see https://www.python-ldap.org/faq.html#usage
            conn.set_option(ldap.OPT_REFERRALS, 0)

        if self.cfg('bindDN'):
            conn.simple_bind_s(self.cfg('bindDN'), self.cfg('bindPassword'))

        try:
            yield conn
        finally:
            conn.unbind_s()


def _as_dict(data):
    d = {}

    for k, v in data.items():
        if not v:
            continue
        if not isinstance(v, list):
            v = [v]
        v = [gws.u.to_str(s) for s in v]
        d[k] = v[0] if len(v) == 1 else v

    return d


def _make_filter(filter_dict):
    conds = ''.join(
        '({}={})'.format(
            ldap.filter.escape_filter_chars(k, 1),
            ldap.filter.escape_filter_chars(v, 1)
        )
        for k, v in filter_dict.items()
    )
    return '(&' + conds + ')'


def _is_member_of(group_dict, user_dn):
    for key in 'member', 'members', 'uniqueMember':
        if key in group_dict and user_dn in group_dict[key]:
            return True


# https://support.microsoft.com/en-us/help/305144

_MS_SCRIPT = 0x0001
_MS_ACCOUNTDISABLE = 0x0002
_MS_HOMEDIR_REQUIRED = 0x0008
_MS_LOCKOUT = 0x0010
_MS_PASSWD_NOTREQD = 0x0020
_MS_PASSWD_CANT_CHANGE = 0x0040
_MS_ENCRYPTED_TEXT_PWD_ALLOWED = 0x0080
_MS_TEMP_DUPLICATE_ACCOUNT = 0x0100
_MS_NORMAL_ACCOUNT = 0x0200
_MS_INTERDOMAIN_TRUST_ACCOUNT = 0x0800
_MS_WORKSTATION_TRUST_ACCOUNT = 0x1000
_MS_SERVER_TRUST_ACCOUNT = 0x2000
_MS_DONT_EXPIRE_PASSWORD = 0x10000
_MS_MNS_LOGON_ACCOUNT = 0x20000
_MS_SMARTCARD_REQUIRED = 0x40000
_MS_TRUSTED_FOR_DELEGATION = 0x80000
_MS_NOT_DELEGATED = 0x100000
_MS_USE_DES_KEY_ONLY = 0x200000
_MS_DONT_REQ_PREAUTH = 0x400000
_MS_PASSWORD_EXPIRED = 0x800000
_MS_TRUSTED_TO_AUTH_FOR_DELEGATION = 0x1000000
_MS_PARTIAL_SECRETS_ACCOUNT = 0x04000000
