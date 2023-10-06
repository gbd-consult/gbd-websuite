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
        self.auth_sql = s.replace(_PLACEHOLDER_LOGIN, '%(login)s').replace(_PLACEHOLDER_PASSWORD, '%(password)s')

        s = self.var('uidSql')
        self.uid_sql = s.replace(_PLACEHOLDER_UID, '%(uid)s')

    def authenticate(self, method: t.IAuthMethod, login, password, **args):
        params = {'login': login, 'password': password}

        with self.db.connect() as conn:
            rs = list(conn.select(self.auth_sql, params))

        if not rs:
            gws.log.info(f'postgres authenticate failed: returned 0 rows')
            return None

        if len(rs) > 1:
            gws.log.error(f'postgres authenticate failed: returned {len(rs)} rows')
            raise gws.common.auth.error.LoginFailed()

        return self._make_user(rs[0], validate=True)

    def get_user(self, user_uid):
        params = {'uid': user_uid}

        with self.db.connect() as conn:
            rs = list(conn.select(self.uid_sql, params))

        if len(rs) != 1:
            gws.log.info(f'postgres get_user failed: returned {len(rs)} rows')
            return None

        return self._make_user(rs[0], validate=False)

    def _make_user(self, rec, validate):
        args = {
            'provider': self,
            'attributes': {}
        }

        valid_user = False
        valid_password = False

        for k, v in rec.items():
            lk = k.lower()

            if lk == 'validuser':
                valid_user = bool(v)
            elif lk == 'validpassword':
                valid_password = bool(v)
            elif lk == 'uid':
                args['uid'] = v
            elif lk == 'roles':
                args['roles'] = gws.as_list(v)
            elif lk == 'displayname':
                args['attributes']['displayName'] = v
            else:
                args['attributes'][k] = v

        if 'uid' not in args:
            raise gws.Error('auth query did not return a uid')

        if validate and not valid_user:
            raise gws.common.auth.error.LoginFailed()

        if validate and not valid_password:
            raise gws.common.auth.error.WrongPassword()
        gws.log.debug(f'LOGIN {args["attributes"]=}')
        return gws.common.auth.user.ValidUser().init_from_source(**args)
