import gws
import gws.base.storage
import gws.lib.date
import gws.lib.sa as sa
import gws.types as t

_DEFAULT_STORE_PATH = gws.MISC_DIR + '/storage8.sqlite'

gws.ext.new.storageProvider('sqlite')


class Config(gws.base.storage.provider.Config):
    """Configuration for sqlite storage."""

    path: t.Optional[str]
    """storage path"""


class Object(gws.base.storage.provider.Object):
    tableName = 'storage'

    dbPath: str
    metaData: sa.MetaData
    engine: sa.Engine
    table: sa.Table

    def configure(self):
        self.dbPath = self.cfg('path', default=_DEFAULT_STORE_PATH)

    def activate(self):
        self.metaData = sa.MetaData()
        self.engine = sa.create_engine(f'sqlite:///{self.dbPath}', echo=True)

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

    def db_list_names(self, cat):
        return sorted(self._exec(sa.select(self.table.c.name)).scalars().all())

    def db_read(self, cat, name):
        stmt = (
            sa.select(self.table)
            .where(self.table.c.category == cat.name)
            .where(self.table.c.name == name))
        for rec in self._exec(stmt).mappings().all():
            return gws.base.storage.provider.Record(
                category=rec['category'],
                name=rec['name'],
                userUid=rec['user_uid'],
                data=rec['data'],
                created=rec['created'],
                updated=rec['updated'],
            )

    def db_write(self, cat, name, data, user_uid):
        rec = self.db_read(cat, name)
        tmp = gws.random_string(64)

        self._exec(sa.insert(self.table).values(
            category=cat.name,
            name=tmp if rec else name,
            user_uid=user_uid,
            data=data,
            created=rec.created if rec else gws.lib.date.timestamp(),
            updated=gws.lib.date.timestamp()
        ))

        if rec:
            self.db_delete(cat, name)
            self._exec(
                sa.update(self.table)
                .where(self.table.c.name == tmp)
                .values(name=name))

    def db_delete(self, cat, name):
        self._exec(
            sa.delete(self.table)
            .where(self.table.c.category == cat.name)
            .where(self.table.c.name == name))

    def _exec(self, stmt):
        with self.engine.begin() as conn:
            return conn.execute(stmt)
