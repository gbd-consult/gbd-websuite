from typing import Optional

import gws
import gws.base.storage
import gws.lib.sqlitex

gws.ext.new.storageProvider('sqlite')


class Config(gws.Config):
    """Configuration for sqlite storage."""

    path: Optional[str]
    """Storage path."""


class Object(gws.StorageProvider):
    dbPath: str
    table = 'storage'

    def configure(self):
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/storage8.sqlite')

    def list_names(self, category):
        rs = self._db().select(f'SELECT name FROM {self.table} WHERE category=:category', category=category)
        return sorted(rec['name'] for rec in rs)

    def read(self, category, name):
        rs = self._db().select(f'SELECT * FROM {self.table} WHERE category=:category AND name=:name', category=category, name=name)
        for rec in rs:
            return gws.StorageRecord(**rec)

    def write(self, category, name, data, user_uid):
        rec = self.read(category, name)
        tmp = gws.u.random_string(64)

        self._db().insert(self.table, dict(
            category=category,
            name=tmp if rec else name,
            user_uid=user_uid,
            data=data,
            created=rec.created if rec else gws.u.stime(),
            updated=gws.u.stime()
        ))

        if rec:
            self.delete(category, name)
            self._db().execute(f'UPDATE {self.table} SET name=:name WHERE name=:tmp', name=name, tmp=tmp)

    def delete(self, category, name):
        self._db().execute(
            f'DELETE FROM {self.table} WHERE category=:category AND name=:name',
            category=category, name=name
        )

    ##

    _sqlitex: gws.lib.sqlitex.Object

    def _db(self):
        if getattr(self, '_sqlitex', None) is None:
            ddl = f'''
                CREATE TABLE IF NOT EXISTS {self.table} (
                    category    TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    user_uid    TEXT,
                    data        TEXT,
                    created     INTEGER,
                    updated     INTEGER,
                    PRIMARY KEY (category, name)
                )
            '''
            self._sqlitex = gws.lib.sqlitex.Object(self.dbPath, ddl)
        return self._sqlitex
