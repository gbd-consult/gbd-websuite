from typing import Optional

import gws.lib.sa as sa

import gws
import gws.base.auth
import gws.lib.datetimex
import gws.lib.jsonx
import gws.lib.osx

gws.ext.new.authSessionManager('sqlite')


class Config(gws.base.auth.session_manager.Config):
    """Configuration for sqlite sessions"""

    path: Optional[str]
    """session storage path"""


# @TODO proper locking


_DEFAULT_TABLE_NAME = 'sessions8.sqlite'


class Object(gws.base.auth.session_manager.Object):
    dbPath: str

    saEngine: sa.Engine
    saTable: sa.Table
    saMetaData: sa.MetaData

    def __getstate__(self):
        return gws.u.omit(vars(self), 'saMetaData', 'saEngine', 'saConnection')

    def configure(self):
        self.dbPath = self.cfg('path', default=gws.c.MISC_DIR + '/' + _DEFAULT_TABLE_NAME)
        self.saMetaData = sa.MetaData()
        self.saTable = sa.Table(
            'sess',
            self.saMetaData,
            sa.Column('uid', sa.String, primary_key=True),
            sa.Column('method_uid', sa.String),
            sa.Column('provider_uid', sa.String),
            sa.Column('user_uid', sa.String),
            sa.Column('str_user', sa.String),
            sa.Column('str_data', sa.String),
            sa.Column('created', sa.Integer),
            sa.Column('updated', sa.Integer),
        )

    def activate(self):
        self.saEngine = sa.create_engine(f'sqlite:///{self.dbPath}')

        if not gws.u.is_file(self.dbPath):
            try:
                self.saMetaData.create_all(self.saEngine, checkfirst=False)
            except sa.exc.SQLAlchemyError as exc:
                raise gws.Error(f'cannot create {self.dbPath!r}') from exc

        # try:
        #     self.saMetaData.reflect(self.saEngine)
        # except sa.exc.SQLAlchemyError as exc:
        #     raise gws.Error(f'cannot open {self.dbPath!r}') from exc

    #

    def cleanup(self):
        self._exec(
            sa.delete(self.saTable).where(self.saTable.c.updated < gws.u.stime() - self.lifeTime))

    def create(self, method, user, data=None):
        am = self.root.app.authMgr
        uid = gws.u.random_string(64)

        self._exec(
            sa.insert(self.saTable).values(
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
        self._exec(
            sa.delete(self.saTable)
            .where(self.saTable.c.uid == sess.uid)
        )

    def delete_all(self):
        self._exec(sa.delete(self.saTable))

    def get(self, uid):
        rs = self._select(
            sa.select(self.saTable)
            .where(self.saTable.c.uid == uid)
        )
        if len(rs) == 0:
            return
        if len(rs) == 1:
            return self._session(rs[0])
        raise gws.Error(f'session_manager: found {len(rs)} sessions for {uid=}')

    def get_valid(self, uid):
        rs = self._select(
            sa.select(self.saTable)
            .where(self.saTable.c.uid == uid)
            .where(self.saTable.c.updated >= gws.u.stime() - self.lifeTime)
        )
        if len(rs) == 0:
            return
        if len(rs) == 1:
            return self._session(rs[0])
        raise gws.Error(f'session_manager: found {len(rs)} sessions for {uid=}')

    def get_all(self):
        return [
            self._session(rec)
            for rec in self._select(sa.select(self.saTable))
        ]

    def save(self, sess):
        if not sess.isChanged:
            return

        self._exec(
            sa.update(self.saTable)
            .where(self.saTable.c.uid == sess.uid)
            .values(
                str_data=gws.lib.jsonx.to_string(sess.data or {}),
                updated=gws.u.stime()
            ))

        sess.isChanged = False

    def touch(self, sess):
        if sess.isChanged:
            return self.save(sess)

        self._exec(
            sa.update(self.saTable)
            .where(self.saTable.c.uid == sess.uid)
            .values(
                updated=gws.u.stime()
            ))

    ##

    def _session(self, rec):
        am = self.root.app.authMgr
        r = gws.u.to_dict(rec)
        return gws.base.auth.session.Object(
            uid=r['uid'],
            method=am.get_method(r['method_uid']),
            user=am.unserialize_user(r['str_user']),
            data=gws.lib.jsonx.from_string(r['str_data']),
            created=gws.lib.datetimex.from_timestamp(r['created']),
            updated=gws.lib.datetimex.from_timestamp(r['updated']),
        )

    def _exec(self, stmt):
        with self.saEngine.begin() as conn:
            conn.execute(stmt)

    def _select(self, stmt):
        with self.saEngine.begin() as conn:
            return list(conn.execute(stmt))
