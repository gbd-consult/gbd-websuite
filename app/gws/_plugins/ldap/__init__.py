import contextlib

import ldap
import ldap.filter

import gws
import gws.types as t
import gws.base.auth.provider
import gws.base.auth.user
import gws.lib.misc as misc
import gws.lib.net

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


class UserSpec(gws.Data):
    """Map LDAP filters to authorization roles"""

    roles: list[str]  #: role names
    matches: t.Optional[str]  #: LDAP filter the account has to match
    memberOf: t.Optional[str]  #: LDAP group the account has to be a member of


class Config(gws.base.auth.provider.Config):
    """LDAP authorization provider"""

    activeDirectory: bool = True  #: true if the LDAP server is ActiveDirectory
    bindDN: str  #: bind DN
    bindPassword: str  #: bind password
    displayNameFormat: gws.FormatStr = '{dn}'  #: format for user's display name
    users: list[UserSpec]  #: map LDAP filters to gws roles
    timeout: gws.Duration = 30  #: LDAP server timeout
    url: str  #: LDAP server url "ldap://host:port/baseDN?searchAttribute"


class Object(gws.base.auth.provider.Object):
    def configure(self):
        

        # the URL is a simplified form of https://httpd.apache.org/docs/2.4/mod/mod_authnz_ldap.html#authldapurl

        p = gws.lib.net.parse_url(self.var('url'))

        self.server = 'ldap://' + p['netloc']
        self.base_dn = p['path'].strip('/')
        self.login_attr = p['query']

        try:
            with self._connection():
                gws.log.debug(f'LDAP connection "{self.uid!r}" is fine')
        except Exception as e:
            raise ValueError(f'LDAP error: {e.__class__.__name__}', *e.args)

    def authenticate(self, method: gws.IAuthMethod, login, password, **args):
        if not password.strip():
            gws.log.warn('empty password, continue')
            return None

        with self._connection() as ld:
            user_data = self._find_user(ld, {self.login_attr: login})
            if not user_data:
                gws.log.warn('user not found, continue')
                return None

            # check for AD disabled accounts
            uac = str(user_data.get('userAccountControl', ''))
            if uac and uac.isdigit():
                if int(uac) & _MS_ACCOUNTDISABLE:
                    gws.log.warn('ACCOUNTDISABLE on, FAIL')
                    raise gws.base.auth.error.AccessDenied()

            try:
                ld.simple_bind_s(user_data['dn'], password)
            except ldap.INVALID_CREDENTIALS:
                gws.log.warn('wrong password, FAIL')
                raise gws.base.auth.error.WrongPassword()
            except ldap.LDAPError:
                gws.log.error('generic fault, FAIL')
                raise gws.base.auth.error.LoginFailed()

            return self._make_user(ld, user_data)

    def get_user(self, user_uid):
        with self._connection() as ld:
            user_data = self._find_user(ld, {self.login_attr: user_uid})

            if not user_data:
                return None

            return self._make_user(ld, user_data)

    @contextlib.contextmanager
    def _connection(self):
        ld = ldap.initialize(self.server)
        ld.set_option(ldap.OPT_NETWORK_TIMEOUT, self.var('timeout'))

        if self.var('activeDirectory'):
            # see https://www.python-ldap.org/faq.html#usage
            ld.set_option(ldap.OPT_REFERRALS, 0)

        if self.var('bindDN'):
            ld.simple_bind_s(
                self.var('bindDN'),
                self.var('bindPassword'))

        try:
            yield ld
        finally:
            ld.unbind_s()

    def _search(self, ld, flt):
        ls = ld.search_s(self.base_dn, ldap.SCOPE_SUBTREE, flt)
        if not ls:
            return
        for dn, data in ls:
            if dn:
                yield dn, data

    def _find_user(self, ld, filter_dict):
        flt = []
        for k, v in filter_dict.items():
            flt.append('(%s=%s)' % (
                ldap.filter.escape_filter_chars(k, 1),
                ldap.filter.escape_filter_chars(v, 1)))
        flt = '(&%s)' % ''.join(flt)

        for dn, data in self._search(ld, flt):
            if dn:
                return _as_dict(dn, data)

    def _find_roles(self, ld, user_data):
        user_dn = user_data['dn']
        roles = set()

        for u in self.var('users'):

            if u.get('matches'):
                for dn, data in self._search(ld, u.matches):
                    if dn == user_dn:
                        roles.update(u.roles)

            elif u.get('memberOf'):
                for dn, data in self._search(ld, u.memberOf):
                    if _is_member_of(_as_dict(dn, data), user_dn):
                        roles.update(u.roles)

        return roles

    def _make_user(self, ld, user_data):
        if 'displayName' not in user_data and self.var('displayNameFormat'):
            user_data['displayName'] = misc.format_placeholders(
                self.var('displayNameFormat'),
                user_data)

        return gws.base.auth.user.ValidUser().init_from_source(
            provider=self,
            uid=user_data[self.login_attr],
            roles=self._find_roles(ld, user_data),
            attributes=user_data)


def _as_dict(dn, data):
    d = {'dn': dn}
    for k, v in data.items():
        if not v:
            continue
        if not isinstance(v, list):
            v = [v]
        v = [gws.as_str(s) for s in v]
        d[k] = v[0] if len(v) == 1 else v
    return d


def _is_member_of(group_data, user_dn):
    for key in 'member', 'members', 'uniqueMember':
        if key in group_data and user_dn in group_data[key]:
            return True
