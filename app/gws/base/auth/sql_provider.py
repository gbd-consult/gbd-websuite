"""Base provider for the sql-based authorization.

SQL-based authentication works by executing SELECT queries against a SQL provider.

The "authorization" query receives the parameters "username", "password", and/or "token" from
an authentication method. If the query doesn't return any rows, the next authentication
provider is attempted. Otherwise, exactly one row should be returned with
at least the following columns:

- ``validuser`` (bool) - mandatory, should be "true" if the user is allowed to log in
- ``validpassword`` (bool) - mandatory, should be "true" if the password is valid
- ``uid`` (str) - user id
- ``roles``(str) - comma-separated list of roles

Column names are case-insensitive.

Other columns, if given, are converted to respective `gws.User` properties.

The "getUser" query receives user ID as a parameter and should return a record for this user.

Example configuration (assuming Postgres with ``pgcrypto``)::

    auth.providers+ {
        type "sql"

        authorizationSql '''
            SELECT
                user.id
                    AS uid,
                user.first_name || ' ' || user.last_name
                    AS displayname,
                user.login
                    AS login,
                user.is_enabled
                    AS validuser,
                ( passwd = crypt({{password}}, passwd) )
                    AS validpassword
            FROM
                public.user
            WHERE
                user.login = {{username}}
        '''

        getUserSql '''
            SELECT
                user.id
                    AS uid,
                user.first_name || ' ' || user.last_name
                    AS displayname,
                user.login
                    AS login
            FROM
                public.user
            WHERE
                user.id = {{uid}}
        '''
    }

"""

from typing import Optional, cast

import re

import gws
import gws.base.auth
import gws.base.database.provider
import gws.config.util
import gws.lib.sa as sa


class Config(gws.base.auth.provider.Config):
    """SQL-based authorization provider"""

    dbUid: Optional[str]
    """Database provider uid"""

    authorizationSql: str
    """Authorization SQL statement"""

    getUserSql: str
    """User data SQL statement"""


class Placeholders(gws.Enum):
    username = 'username'
    password = 'password'
    token = 'token'
    uid = 'uid'


class Object(gws.base.auth.provider.Object):
    db: gws.DatabaseProvider
    authorizationSql: str
    getUserSql: str

    def configure(self):
        self.configure_provider()
        self.authorizationSql = self.cfg('authorizationSql')
        self.getUserSql = self.cfg('getUserSql')

    def configure_provider(self):
        return gws.config.util.configure_database_provider_for(self)

    def authenticate(self, method, credentials):
        params = {
            Placeholders.username: credentials.get('username'),
            Placeholders.password: credentials.get('password'),
            Placeholders.token: credentials.get('token'),
        }

        rs = self._get_records(self.authorizationSql, params)

        if not rs:
            return
        if len(rs) > 1:
            raise gws.ForbiddenError(f'multiple records found')

        return self._make_user(rs[0], validate=True)

    def get_user(self, local_uid):
        params = {
            'uid': local_uid,
        }

        rs = self._get_records(self.getUserSql, params)

        if not rs:
            return
        if len(rs) > 1:
            return

        return self._make_user(rs[0], validate=False)

    def _get_records(self, sql: str, params: dict) -> list[dict]:
        sql = re.sub(r'{(\w+)}', r':\1', sql)
        with self.db.connect() as conn:
            return [gws.u.to_dict(r) for r in conn.execute(sa.text(sql), params)]

    def _make_user(self, rec: dict, validate: bool) -> gws.User:
        user_rec = {}

        valid_user = False
        valid_password = False

        for k, v in rec.items():
            lk = k.lower()

            if lk == 'validuser':
                valid_user = bool(v)
            elif lk == 'validpassword':
                valid_password = bool(v)
            elif lk == 'uid':
                user_rec['localUid'] = str(v)
            elif lk == 'roles':
                user_rec['roles'] = set(map(str.strip, v.split(',')))
            else:
                user_rec[k] = v

        if 'localUid' not in user_rec:
            raise gws.ForbiddenError('no uid returned')

        if validate and not valid_user:
            raise gws.ForbiddenError(f'invalid user')

        if validate and not valid_password:
            raise gws.ForbiddenError(f'invalid password')

        return gws.base.auth.user.from_record(self, user_rec)
