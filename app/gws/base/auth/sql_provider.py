"""Base provider for the sql-based authorization.

SQL-based authentication works by executing SELECT queries against a SQL provider.

The "authorization" query receives the parameters "username", "password", and/or "token" from
an authentication method. If the query doesn't return any rows, the next authentication
provider is attempted. Otherwise, exactly one row should be returned with
at least the following columns:

- ``validuser`` (bool) - mandatory, should be "true" if the user is allowed to log in
- ``validpassword`` (bool) - mandatory, should be "true" if the password is valid
- ``uid`` - mandatory, user id
- ``roles`` - optional, comma-separated list of roles
- ``displayname`` - optional, user's display name
- ``login`` - optional, user's login name

If more columns are returned, they become attributes of the User object and
can be used for templating.

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


class Columns(gws.Enum):
    validuser = 'validuser'
    validpassword = 'validpassword'
    uid = 'uid'
    roles = 'roles'
    displayname = 'displayname'
    login = 'login'


class Object(gws.base.auth.provider.Object):
    dbProvider: gws.DatabaseProvider
    authorizationSql: str
    getUserSql: str

    def configure(self):
        self.uid = 'gws.base.auth.providers.sql'
        self.dbProvider = cast(gws.DatabaseProvider, gws.base.database.provider.get_for(self))
        self.authorizationSql = self.cfg('authorizationSql')
        self.getUserSql = self.cfg('getUserSql')

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

        with self.dbProvider.connection() as conn:
            stmt = sa.text(sql)
            return [r._asdict() for r in conn.execute(stmt, params)]

    def _make_user(self, rec: dict, validate: bool) -> gws.User:
        args = {
            'attributes': {}
        }

        valid_user = False
        valid_password = False

        for k, v in rec.items():
            lk = k.lower()

            if lk == Columns.validuser:
                valid_user = bool(v)
            elif lk == Columns.validpassword:
                valid_password = bool(v)
            elif lk == Columns.uid:
                args['localUid'] = str(v)
            elif lk == Columns.roles:
                args['roles'] = gws.u.to_list(v)
            elif lk == Columns.displayname:
                args['displayName'] = v
            elif lk == Columns.login:
                args['loginName'] = v
            else:
                args['attributes'][k] = v

        if 'localUid' not in args:
            raise gws.ForbiddenError('no uid returned')

        if validate and not valid_user:
            raise gws.ForbiddenError(f'invalid user')

        if validate and not valid_password:
            raise gws.ForbiddenError(f'invalid password')

        return gws.base.auth.user.init(provider=self, **args)
