"""Provider for the Postgres authorization"""

import json

import gws
import gws.common.auth
import gws.common.auth.provider
import gws.common.auth.user
import gws.common.db
import gws.types as t


class Config(gws.common.auth.provider.Config):
    """Postgres authorization provider"""

    db: str = ''  #: database provider ID
    authSql: str  #: sql statement for login check
    uidSql: str  #: sql statement for fetching user data


_PLACEHOLDER_LOGIN = '{login}'
_PLACEHOLDER_PASSWORD = '{password}'
_PLACEHOLDER_UID = '{uid}'


class Object(gws.common.auth.provider.Object):
    def configure(self):
        super().configure()

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        s = self.var('authSql')

        try:
            a = s.index(_PLACEHOLDER_LOGIN)
            b = s.index(_PLACEHOLDER_PASSWORD)
        except ValueError:
            raise gws.Error('invalid "authSql"')

        self.auth_param_order = 'lp' if a < b else 'pl'
        self.auth_sql = s.replace(_PLACEHOLDER_LOGIN, '%s').replace(_PLACEHOLDER_PASSWORD, '%s')

        s = self.var('uidSql')
        if _PLACEHOLDER_UID not in s:
            raise gws.Error('invalid "uidSql"')

        self.uid_sql = s.replace(_PLACEHOLDER_UID, '%s')

    def authenticate(self, method: t.IAuthMethod, login, password, **args):
        params = [login, password]
        if self.auth_param_order == 'pl':
            params = [password, login]

        with self.db.connect() as conn:
            rs = list(conn.select(self.auth_sql, params))

        if len(rs) == 1:
            return self._make_user(rs[0])

        gws.log.info(f'postgres authenticate failed: returned {len(rs)} rows')

    def get_user(self, user_uid):
        with self.db.connect() as conn:
            rs = list(conn.select(self.uid_sql, [user_uid]))

        if len(rs) == 1:
            return self._make_user(rs[0])

        gws.log.info(f'postgres get_user failed: returned {len(rs)} rows')

    def _make_user(self, rec):
        args = {
            'provider': self,
            'attributes': {}
        }
        for k, v in rec.items():
            if k.lower() == 'uid':
                args['uid'] = v
            elif k.lower() == 'roles':
                args['roles'] = gws.as_list(v)
            elif k.lower() == 'displayName':
                args['attributes']['displayName'] = v
            else:
                args['attributes'][k] = v

        if 'uid' not in args:
            raise gws.Error('auth query did not return a uid')

        return gws.common.auth.user.ValidUser().init_from_source(**args)
