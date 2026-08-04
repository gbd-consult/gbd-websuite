from typing import Optional

import gws
import gws.base.auth
import gws.lib.datetimex as dtx
import gws.lib.jsonx
import gws.lib.sqlitex

gws.ext.new.authSessionManager('sqlite')

_CLEANUP_INTERVAL = 600
_TOUCH_INTERVAL = 60


class Config(gws.base.auth.session_manager.Config):
    """Configuration for sqlite sessions"""

    path: Optional[str]
    """Session storage path."""


class Object(gws.base.auth.session_manager.Object):
    dbPath: str
    table = 'sessions'

    def configure(self):
        ver = self.root.specs.version.rpartition('.')[0]
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/sessions.{ver}.sqlite')

    ##

    _cleanupTime = 0

    def cleanup(self):
        self._db().execute(
            f'DELETE FROM {self.table} WHERE updated < :last_time OR created < :max_time',
            last_time=gws.u.stime() - self.lifeTime,
            max_time=gws.u.stime() - self.maxLifeTime,
        )
        self._cleanupTime = gws.u.stime()

    def create(self, method, user, data=None):
        if gws.u.stime() > self._cleanupTime + _CLEANUP_INTERVAL:
            self.cleanup()

        am = self.root.app.authMgr
        uid = gws.u.random_string(64)

        self._db().insert(self.table, dict(
            uid=uid,
            method_uid=method.uid,
            user_uid=user.uid,
            str_user=am.serialize_user(user),
            str_data=gws.lib.jsonx.to_string(data or {}),
            created=gws.u.stime(),
            updated=gws.u.stime(),
        ))

        return self.get(uid)

    def delete(self, sess):
        self._db().execute(f'DELETE FROM {self.table} WHERE uid=:uid', uid=sess.uid)

    def delete_all(self):
        self._db().execute(f'DELETE FROM {self.table}')

    def get(self, uid):
        rs = self._db().select(
            f'SELECT * FROM {self.table} WHERE uid=:uid AND updated >= :last_time AND created >= :max_time',
            uid=uid,
            last_time=gws.u.stime() - self.lifeTime,
            max_time=gws.u.stime() - self.maxLifeTime,
        )
        if len(rs) == 1:
            return self._session(rs[0])

    def list_all(self):
        return [
            self._session(rec)
            for rec in self._db().select(f'SELECT * FROM {self.table}')
        ]

    def save(self, sess):
        if not sess.isChanged:
            return

        self._db().execute(
            f'UPDATE {self.table} SET str_data=:str_data, updated=:updated WHERE uid=:uid',
            str_data=gws.lib.jsonx.to_string(sess.data or {}),
            updated=gws.u.stime(),
            uid=sess.uid
        )

        sess.isChanged = False

    def touch(self, sess):
        if sess.isChanged:
            return self.save(sess)

        # throttle updates to avoid excessive writes on high-traffic sessions
        if gws.u.stime() - dtx.to_timestamp(sess.updated) < _TOUCH_INTERVAL:
            return

        self._db().execute(
            f'UPDATE {self.table} SET updated=:updated WHERE uid=:uid',
            updated=gws.u.stime(),
            uid=sess.uid
        )

    ##

    def _session(self, rec):
        am = self.root.app.authMgr
        r = gws.u.to_dict(rec)
        usr = am.unserialize_user(r['str_user'])
        if not usr:
            gws.log.error(f'invalid user in session {r["uid"]!r}')
            self._db().execute(f'DELETE FROM {self.table} WHERE uid=:uid', uid=r['uid'])
            usr = am.guestUser
        return gws.base.auth.session.Object(
            uid=r['uid'],
            method=am.get_method(r['method_uid']),
            user=usr,
            data=gws.lib.jsonx.from_string(r['str_data']),
            created=dtx.from_timestamp(r['created']),
            updated=dtx.from_timestamp(r['updated']),
        )

    ##

    _sqlitex: gws.lib.sqlitex.Object

    def _db(self):
        if getattr(self, '_sqlitex', None) is None:
            ddl = f'''
                CREATE TABLE IF NOT EXISTS {self.table} (
                    uid          TEXT NOT NULL PRIMARY KEY,
                    method_uid   TEXT NOT NULL,
                    user_uid     TEXT NOT NULL,
                    str_user     TEXT NOT NULL,
                    str_data     TEXT NOT NULL,
                    created      INTEGER NOT NULL,
                    updated      INTEGER NOT NULL
                )
            '''
            self._sqlitex = gws.lib.sqlitex.Object(self.dbPath, ddl)
        return self._sqlitex
