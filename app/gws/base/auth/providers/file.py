"""Provider for the file-based authorization"""

import getpass

import gws
import gws.lib.json2
import gws.lib.password
import gws.types as t

from .. import error, provider, user


@gws.ext.config.authProvider('file')
class Config(provider.Config):
    """File-based authorization provider"""

    path: gws.FilePath  #: path to the users json file


@gws.ext.object.authProvider('file')
class Object(provider.Object):
    path: str
    db: t.List[dict]

    def configure(self):
        self.path = self.var('path')
        self.db = gws.lib.json2.from_path(self.path)

    def authenticate(self, method, credentials):
        wrong_password = 0
        found = []

        username = credentials.get('username')
        if not username:
            return

        for rec in self.db:
            login_ok = gws.lib.password.cmp(username, rec['login'])
            password_ok = gws.lib.password.check(credentials.get('password'), rec['password'])
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
        return user.create(
            user.AuthorizedUser,
            provider=self,
            local_uid=rec['login'],
            roles=rec.get('roles', []),
            attributes={'displayName': rec.get('name', rec['login'])}
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
