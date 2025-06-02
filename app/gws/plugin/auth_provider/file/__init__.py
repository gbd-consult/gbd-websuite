"""Provider for the file-based authorization.

This provider works with a local JSON file, which is expected to contain
a list of user "records" (dicts).

A record is required to contain fields ``login`` and ``password`` (hashed as per `gws.lib.password.encode`).

Other fields, if given, are converted to respective `gws.User` properties.
"""

import getpass

import gws
import gws.base.auth
import gws.lib.jsonx
import gws.lib.password

gws.ext.new.authProvider('file')


class Config(gws.base.auth.provider.Config):
    """File-based authorization provider."""

    path: gws.FilePath
    """Path to the users json file."""


class Object(gws.base.auth.provider.Object):
    path: str
    db: list[dict]

    def configure(self):
        self.path = self.cfg('path')
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
            raise gws.ForbiddenError(f'wrong password for {username!r}')

        if len(found) > 1:
            raise gws.ForbiddenError(f'multiple entries for {username!r}')

        if len(found) == 1:
            return self._make_user(found[0])

    def get_user(self, local_uid):
        for rec in self.db:
            if rec['login'] == local_uid:
                return self._make_user(rec)

    def _make_user(self, rec: dict):
        user_rec = dict(rec)

        login = user_rec.pop('login', '')
        user_rec['localUid'] = user_rec['loginName'] = login
        user_rec['displayName'] = user_rec.pop('name', login)
        user_rec.pop('password', '')

        return gws.base.auth.user.from_record(self, user_rec)

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
