"""Provider for the file-based authorization"""

import getpass

import gws
import gws.base.auth.error
import gws.base.auth.provider
import gws.base.auth.user
import gws.lib.jsonx
import gws.lib.password
import gws.types as t

gws.ext.new.authProvider('file')


class Config(gws.base.auth.provider.Config):
    """File-based authorization provider"""

    path: gws.FilePath
    """path to the users json file"""


class Object(gws.base.auth.provider.Object):
    path: str
    db: t.List[dict]

    def configure(self):
        self.uid = 'gws.base.auth.providers.file'
        self.path = self.var('path')
        self.db = gws.lib.jsonx.from_path(self.path)

    def authenticate(self, method, credentials):
        wrong_password = 0
        found = []

        username = credentials.get('username')
        password = credentials.get('password')
        if not username or not password:
            return

        for rec in self.db:
            login_ok = gws.lib.password.compare(username, rec['login'])
            password_ok = gws.lib.password.check(password, rec['password'])
            if login_ok and password_ok:
                found.append(rec)
            if login_ok and not password_ok:
                wrong_password += 1

        if wrong_password:
            gws.log.error(f'wrong password for {username!r}')
            return

        if len(found) > 1:
            gws.log.error(f'multiple entries for {username!r}')
            return

        if len(found) == 1:
            return self._make_user(found[0])

    def get_user(self, local_uid):
        for rec in self.db:
            if rec['login'] == local_uid:
                return self._make_user(rec)

    def _make_user(self, rec):
        atts = dict(rec)
        login = atts.pop('login')
        atts.pop('password', '')

        return gws.base.auth.user.from_args(
            gws.base.auth.user.AuthorizedUser,
            provider=self,
            displayName=atts.pop('name', login),
            localUid=login,
            loginName=login,
            roles=atts.pop('roles', []),
            pendingMfa=atts.pop('mfa', None),
            attributes=atts,
        )

    @gws.ext.command.cli('authPassword')
    def passwd(self, p: gws.EmptyRequest):
        """Encode a password for the authorization file"""

        while True:
            p1 = getpass.getpass('Password: ')
            p2 = getpass.getpass('Repeat  : ')

            if p1 != p2:
                print('passwords do not match')
                continue

            p = gws.lib.password.encode(p1)
            print(p)
            break
