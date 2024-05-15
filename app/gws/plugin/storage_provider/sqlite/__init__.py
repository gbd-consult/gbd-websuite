from typing import Optional

import gws
import gws.base.storage
import gws.lib.sa as sa

gws.ext.new.storageProvider('sqlite')


class Config(gws.Config):
    """Configuration for sqlite storage."""

    path: Optional[str]
    """storage path"""


class Object(gws.StorageProvider):
    tableName = 'storage'

    dbPath: str
    metaData: sa.MetaData
    engine: sa.Engine
    table: sa.Table

    def configure(self):
        _DEFAULT_STORE_PATH = gws.c.MISC_DIR + '/storage8.sqlite'
        self.dbPath = self.cfg('path', default=_DEFAULT_STORE_PATH)

    def activate(self):
        self.metaData = sa.MetaData()
        self.engine = sa.create_engine(f'sqlite:///{self.dbPath}')

        self.table = sa.Table(
            'storage',
            self.metaData,
            sa.Column('category', sa.String, primary_key=True),
            sa.Column('name', sa.String, primary_key=True),
            sa.Column('user_uid', sa.String),
            sa.Column('data', sa.String),
            sa.Column('created', sa.Integer),
            sa.Column('updated', sa.Integer),
        )

        try:
            self.metaData.create_all(self.engine, checkfirst=True)
        except Exception as exc:
            raise gws.Error(f'cannot open {self.dbPath!r}') from exc

    def list_names(self, category):
        stmt = sa.select(self.table.c.name).where(self.table.c.category == category)
        return sorted(self._exec(stmt).scalars().all())

    def read(self, category, name):
        stmt = sa.select(self.table).where(self.table.c.category == category).where(self.table.c.name == name)
        for rec in self._exec(stmt).mappings().all():
            return gws.StorageRecord(
                category=rec['category'],
                name=rec['name'],
                userUid=rec['user_uid'],
                data=rec['data'],
                created=rec['created'],
                updated=rec['updated'],
            )

    def write(self, category, name, data, user_uid):
        rec = self.read(category, name)
        tmp = gws.u.random_string(64)

        self._exec(sa.insert(self.table).values(
            category=category,
            name=tmp if rec else name,
            user_uid=user_uid,
            data=data,
            created=rec.created if rec else gws.u.stime(),
            updated=gws.u.stime()
        ))

        if rec:
            self.delete(category, name)
            self._exec(
                sa.update(self.table)
                .where(self.table.c.name == tmp)
                .values(name=name))

    def delete(self, category, name):
        self._exec(
            sa.delete(self.table)
            .where(self.table.c.category == category)
            .where(self.table.c.name == name))

    def _exec(self, stmt):
        with self.engine.begin() as conn:
            return conn.execute(stmt)
