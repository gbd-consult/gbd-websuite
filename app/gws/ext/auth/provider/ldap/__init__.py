import ldap
import ldap.filter
import urllib.parse
import contextlib

import gws
import gws.auth.provider
import gws.auth.error
import gws.auth.user
import gws.tools.misc as misc

import gws.types as t

# https://support.microsoft.com/en-us/help/305144

MS_SCRIPT = 0x0001
MS_ACCOUNTDISABLE = 0x0002
MS_HOMEDIR_REQUIRED = 0x0008
MS_LOCKOUT = 0x0010
MS_PASSWD_NOTREQD = 0x0020
MS_PASSWD_CANT_CHANGE = 0x0040
MS_ENCRYPTED_TEXT_PWD_ALLOWED = 0x0080
MS_TEMP_DUPLICATE_ACCOUNT = 0x0100
MS_NORMAL_ACCOUNT = 0x0200
MS_INTERDOMAIN_TRUST_ACCOUNT = 0x0800
MS_WORKSTATION_TRUST_ACCOUNT = 0x1000
MS_SERVER_TRUST_ACCOUNT = 0x2000
MS_DONT_EXPIRE_PASSWORD = 0x10000
MS_MNS_LOGON_ACCOUNT = 0x20000
MS_SMARTCARD_REQUIRED = 0x40000
MS_TRUSTED_FOR_DELEGATION = 0x80000
MS_NOT_DELEGATED = 0x100000
MS_USE_DES_KEY_ONLY = 0x200000
MS_DONT_REQ_PREAUTH = 0x400000
MS_PASSWORD_EXPIRED = 0x800000
MS_TRUSTED_TO_AUTH_FOR_DELEGATION = 0x1000000
MS_PARTIAL_SECRETS_ACCOUNT = 0x04000000


class RoleSpec(t.Data):
    """map authorization roles to LDAP filters"""

    role: str  #: gws role name
    matches: t.Optional[str]  #: LDAP filter the account has to match
    memberOf: t.Optional[str]  #: LDAP group the account has to be a member of


class Config(t.WithType):
    """LDAP authorization provider"""

    activeDirectory: bool = True  #: true if the LDAP server is ActiveDirectory
    bindDN: str  #: bind DN
    bindPassword: str  #: bind password
    displayNameFormat: t.formatstr = '{dn}'  #: format for user's display name
    roles: t.List[RoleSpec]  #: map LDAP filters to roles
    timeout: t.duration = 30  #: LDAP server timeout
    url: str  #: LDAP server url "ldap://host:port/baseDN?searchAttribute"
    uid: t.Optional[str]  #: unique id


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


class Object(gws.auth.provider.Object):
    server = ''
    base = ''
    loginProp = ''

    def configure(self):
        super().configure()

        p = urllib.parse.urlparse(self.var('url'))
        self.server = 'ldap://' + p.netloc
        self.base = p.path.strip('/')
        self.loginProp = p.query

        try:
            with self._connection():
                gws.log.info(f'LDAP connection "{self.uid}" is fine')
        except Exception as e:
            raise ValueError('LDAP error: %s' % e.__class__.__name__, *e.args)

    def authenticate_user(self, login, password, **args):
        if not password.strip():
            gws.log.warn('empty password, continue')
            return None

        with self._connection() as ld:
            user_data = self._find_user(ld, {self.loginProp: login})
            if not user_data:
                gws.log.warn('user not found, continue')
                return None

            # check for AD disabled accounts
            uac = str(user_data.get('userAccountControl', ''))
            if uac and uac.isdigit():
                if int(uac) & MS_ACCOUNTDISABLE:
                    gws.log.warn('ACCOUNTDISABLE on, FAIL')
                    raise gws.auth.error.AccessDenied()

            try:
                ld.simple_bind_s(user_data['dn'], password)
            except ldap.INVALID_CREDENTIALS:
                gws.log.warn('wrong password, FAIL')
                raise gws.auth.error.WrongPassword()
            except ldap.LDAPError:
                gws.log.error('generic fault, FAIL')
                raise gws.auth.error.LoginFailed()

            return self._make_user(ld, user_data)

    def get_user(self, user_uid):
        with self._connection() as ld:
            user_data = self._find_user(ld, {self.loginProp: user_uid})

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
        ls = ld.search_s(self.base, ldap.SCOPE_SUBTREE, flt)
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

        for r in self.var('roles'):

            if r.get('matches'):
                for dn, data in self._search(ld, r.matches):
                    if dn == user_dn:
                        yield r.role

            elif r.get('memberOf'):
                for dn, data in self._search(ld, r.memberOf):
                    if _is_member_of(_as_dict(dn, data), user_dn):
                        yield r.role

            else:
                yield r.role

    def _make_user(self, ld, user_data):
        if 'displayName' not in user_data and self.var('displayNameFormat'):
            user_data['displayName'] = misc.format_placeholders(
                self.var('displayNameFormat'),
                user_data)

        roles = set(self._find_roles(ld, user_data))
        return self.root.create(gws.auth.user.ValidUser).init_from_source(
            provider=self,
            uid=user_data[self.loginProp],
            roles=roles,
            attributes=user_data)
