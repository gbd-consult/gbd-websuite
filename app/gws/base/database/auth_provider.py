"""Base authorization provider that checks users with SQL queries."""

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
    """UID of the database provider."""

    authorizationSql: str
    """SQL query that checks user credentials."""

    getUserSql: str
    """SQL query that returns the user record for the {uid} placeholder."""


class Placeholders(gws.Enum):
    """Placeholder names available in the SQL queries."""

    username = 'username'
    """User name from the credentials."""
    password = 'password'
    """Password from the credentials."""
    token = 'token'
    """Token from the credentials."""
    uid = 'uid'
    """Local user id, used in ``getUserSql``."""


class Object(gws.base.auth.provider.Object):
    """Authorization provider that checks credentials and loads users with SQL queries."""

    db: gws.DatabaseProvider
    """Database provider the queries run on."""
    authorizationSql: str
    """SQL query that checks user credentials."""
    getUserSql: str
    """SQL query that returns the record of a user by uid."""

    def configure(self):
        self.configure_provider()
        self.authorizationSql = self.cfg('authorizationSql')
        self.getUserSql = self.cfg('getUserSql')

    def configure_provider(self):
        """Set the database provider from the configuration.

        Returns:
            ``True`` if a provider was found.

        Raises:
            ``gws.Error``: If no matching database provider is configured.
        """
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
        """Run a query with ``{name}`` placeholders converted to bind parameters."""
        sql = re.sub(r'{(\w+)}', r':\1', sql)
        with self.db.begin() as conn:
            return [gws.u.to_dict(r) for r in conn.execute(sa.text(sql), params)]

    def _make_user(self, rec: dict, validate: bool) -> gws.User:
        """Create a user from a record, checking ``validuser`` and ``validpassword`` if ``validate`` is set."""
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
            else:
                user_rec[k] = v

        if 'localUid' not in user_rec:
            raise gws.ForbiddenError('no uid returned')

        if validate and not valid_user:
            raise gws.ForbiddenError(f'invalid user')

        if validate and not valid_password:
            raise gws.ForbiddenError(f'invalid password')

        return gws.base.auth.user.from_record(self, user_rec)
