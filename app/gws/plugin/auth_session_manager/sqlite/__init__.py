import gws.lib.sa as sa

import gws
import gws.base.auth
import gws.lib.date
import gws.lib.jsonx
import gws.lib.osx

import gws.types as t

gws.ext.new.authSessionManager('sqlite')


class Config(gws.base.auth.session_manager.Config):
    """Configuration for sqlite sessions"""

    path: t.Optional[str]
    """session storage path"""


# @TODO proper locking

class Object(gws.base.auth.session_manager.Object):
    dbPath: str
    metaData: sa.MetaData
    engine: sa.Engine
    table: sa.Table

    def configure(self):
        _DEFAULT_STORE_PATH = gws.MISC_DIR + '/sessions8.sqlite'
        self.dbPath = self.cfg('path', default=_DEFAULT_STORE_PATH)

    def activate(self):
        self.metaData = sa.MetaData()
        self.engine = sa.create_engine(f'sqlite:///{self.dbPath}')

        self.table = sa.Table(
            'sess',
            self.metaData,
            sa.Column('uid', sa.String, primary_key=True),
            sa.Column('method_uid', sa.String),
            sa.Column('provider_uid', sa.String),
            sa.Column('user_uid', sa.String),
            sa.Column('str_user', sa.String),
            sa.Column('str_data', sa.String),
            sa.Column('created', sa.Integer),
            sa.Column('updated', sa.Integer),
        )

        if not gws.is_file(self.dbPath):
            try:
                self.metaData.create_all(self.engine, checkfirst=False)
            except sa.exc.SQLAlchemyError as exc:
                raise gws.Error(f'cannot create {self.dbPath!r}') from exc

        try:
            self.metaData.reflect(self.engine)
        except sa.exc.SQLAlchemyError as exc:
            raise gws.Error(f'cannot open {self.dbPath!r}') from exc

    #

    def cleanup(self):
        self._exec(
            sa.delete(self.table).where(self.table.c.updated < gws.lib.date.timestamp() - self.lifeTime))

    def create(self, method, user, data=None):
        am = self.root.app.authMgr
        uid = gws.random_string(64)

        self._exec(sa.insert(self.table).values(
            uid=uid,
            method_uid=method.uid,
            user_uid=user.uid,
            str_user=am.serialize_user(user),
            str_data=gws.lib.jsonx.to_string(data or {}),
            created=gws.lib.date.timestamp(),
            updated=gws.lib.date.timestamp(),
        ))

        return self.get(uid)

    def delete(self, sess):
        self._exec(
            sa.delete(self.table).where(self.table.c.uid == sess.uid))

    def delete_all(self):
        self.metaData.create_all(self.engine, checkfirst=False)

    def get(self, uid):
        stmt = sa.select(self.table).where(self.table.c.uid == uid)
        with self.engine.begin() as conn:
            for rec in conn.execute(stmt).mappings().all():
                return self._session(rec)

    def get_valid(self, uid):
        stmt = (
            sa
            .select(self.table)
            .where(self.table.c.uid == uid)
            .where(self.table.c.updated >= gws.lib.date.timestamp() - self.lifeTime))

        with self.engine.begin() as conn:
            for rec in conn.execute(stmt).mappings().all():
                return self._session(rec)

    def get_all(self):
        stmt = sa.select(self.table)
        with self.engine.begin() as conn:
            return [
                self._session(rec)
                for rec in conn.execute(stmt).mappings().all()
            ]

    def save(self, sess):
        if not sess.isChanged:
            return

        self._exec(
            sa.update(self.table).where(self.table.c.uid == sess.uid).values(
                str_data=gws.lib.jsonx.to_string(sess.data or {}),
                updated=gws.lib.date.timestamp()))

        sess.isChanged = False

    def touch(self, sess):
        if sess.isChanged:
            return self.save(sess)

        self._exec(sa.update(self.table).where(self.table.c.uid == sess.uid).values(
            updated=gws.lib.date.timestamp()))

    ##

    def _session(self, rec):
        am = self.root.app.authMgr
        return gws.base.auth.session.Object(
            uid=rec['uid'],
            method=am.get_method(rec['method_uid']),
            user=am.unserialize_user(rec['str_user']),
            data=gws.lib.jsonx.from_string(rec['str_data']),
            created=gws.lib.date.from_timestamp(rec['created']),
            updated=gws.lib.date.from_timestamp(rec['updated']),
        )

    def _exec(self, stmt):
        with self.engine.begin() as conn:
            conn.execute(stmt)
