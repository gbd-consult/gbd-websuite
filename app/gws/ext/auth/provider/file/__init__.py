"""Provider for the file-based authorization"""

import json

import gws
import gws.auth.error
import gws.auth.provider
import gws.auth.user
import gws.tools.password
import gws.types as t


class Config(t.WithType):
    """file-based authorization provider"""

    path: t.filepath  #: path to the users json file


class Object(gws.auth.provider.Object):
    def authenticate_user(self, login, password, **args):
        db = _read(self.var('path'))

        ls = [
            gws.tools.password.cmp(login, rec['login']) * 2 + gws.tools.password.check(password, rec['password'])
            for rec in db
        ]

        if any(x == 2 for x in ls):
            raise gws.auth.error.WrongPassword()

        if any(x == 3 for x in ls):
            return self.get_user(login)

    def get_user(self, user_uid):
        db = _read(self.var('path'))
        for rec in db:
            if rec['login'] == user_uid:
                return self._make_user(rec)

    def _make_user(self, rec):
        return self.root.create(gws.auth.user.ValidUser).init_from_source(
            provider=self,
            uid=rec['login'],
            roles=rec.get('roles', []),
            attributes={'displayName': rec.get('name', rec['login'])}
        )


def _read(path):
    try:
        with open(path, encoding='utf8') as fp:
            return json.load(fp)
    except IOError:
        return {}
