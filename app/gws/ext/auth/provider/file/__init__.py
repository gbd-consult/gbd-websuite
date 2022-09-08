"""Provider for the file-based authorization"""

import json

import gws
import gws.common.auth
import gws.common.auth.provider
import gws.common.auth.user
import gws.tools.password
import gws.types as t


class Config(gws.common.auth.provider.Config):
    """File-based authorization provider"""

    path: t.FilePath  #: path to the users json file


class Object(gws.common.auth.provider.Object):
    def configure(self):
        super().configure()
        self.path = self.var('path')

    def authenticate(self, method: t.IAuthMethod, login, password, **args):
        wrong_password = 0
        found = []

        for rec in self._db():
            login_ok = gws.tools.password.cmp(login, rec['login'])
            password_ok = gws.tools.password.check(password, rec['password'])
            if login_ok and password_ok:
                found.append(rec)
            if login_ok and not password_ok:
                wrong_password += 1

        if wrong_password:
            raise gws.common.auth.error.WrongPassword()

        if len(found) == 1:
            return self._make_user(found[0])

    def get_user(self, user_uid):
        for rec in self._db():
            if rec['login'] == user_uid:
                return self._make_user(rec)

    def _make_user(self, rec):
        return gws.common.auth.user.ValidUser().init_from_source(
            provider=self,
            uid=rec['login'],
            roles=rec.get('roles', []),
            attributes={'displayName': rec.get('name', rec['login'])}
        )

    def _db(self):
        try:
            with open(self.path, encoding='utf8') as fp:
                return json.load(fp)
        except IOError:
            return []
