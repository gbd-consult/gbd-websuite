"""Provider for the file-based authorization"""

import json

import gws
import gws.common.auth
import gws.common.auth.provider
import gws.common.auth.user
import gws.tools.password
import gws.types as t


class Config(t.WithType):
    """File-based authorization provider"""

    path: t.FilePath  #: path to the users json file


class Object(gws.common.auth.provider.Object):
    def __init__(self):
        super().__init__()
        self.path = ''

    def configure(self):
        super().configure()
        self.path = self.var('path')

    def authenticate(self, login, password, **args):
        db = self._read()

        ls = [
            gws.tools.password.cmp(login, rec['login']) * 2 + gws.tools.password.check(password, rec['password'])
            for rec in db
        ]

        if any(x == 2 for x in ls):
            raise gws.common.auth.error.WrongPassword()

        if any(x == 3 for x in ls):
            return self.get_user(login)

    def get_user(self, user_uid):
        db = self._read()

        for rec in db:
            if rec['login'] == user_uid:
                return self._make_user(rec)

    def _make_user(self, rec):
        return gws.common.auth.user.ValidUser().init_from_source(
            provider=self,
            uid=rec['login'],
            roles=rec.get('roles', []),
            attributes={'displayName': rec.get('name', rec['login'])}
        )

    def _read(self):
        try:
            with open(self.path, encoding='utf8') as fp:
                return json.load(fp)
        except IOError:
            return {}
