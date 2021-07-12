"""Provider for the file-based authorization"""

import getpass

import gws
import gws.lib.json2
import gws.lib.password
from .. import core, error


@gws.ext.Config('auth.provider.file')
class Config(core.ProviderConfig):
    """File-based authorization provider"""

    path: gws.FilePath  #: path to the users json file


@gws.ext.Object('auth.provider.file')
class Object(core.Provider):
    path: str

    def configure(self):
        self.path = self.var('path')

    def authenticate(self, method, credentials):
        wrong_password = 0
        found = []

        for rec in self._db():
            login_ok = gws.lib.password.cmp(credentials.get('username'), rec['login'])
            password_ok = gws.lib.password.check(credentials.get('password'), rec['password'])
            if login_ok and password_ok:
                found.append(rec)
            if login_ok and not password_ok:
                wrong_password += 1

        if wrong_password:
            raise error.WrongPassword()

        if len(found) == 1:
            return self._make_user(found[0])

    def get_user(self, user_uid):
        for rec in self._db():
            if rec['login'] == user_uid:
                return self._make_user(rec)

    def _make_user(self, rec):
        return core.ValidUser().init_from_source(
            provider=self,
            uid=rec['login'],
            roles=rec.get('roles', []),
            attributes={'displayName': rec.get('name', rec['login'])}
        )

    def _db(self):
        return gws.lib.json2.from_path(self.path)

    @gws.ext.command('cli.auth.password')
    def passwd(self, p: gws.NoParams):
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


