"""Postgres storage provider."""

from typing import Optional

import gws
import gws.config.util
import gws.lib.sa as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from . import provider

gws.ext.new.storageProvider('postgres')


TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        category    TEXT NOT NULL,
        name        TEXT NOT NULL,
        user_uid    TEXT,
        data        TEXT,
        created     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (category, name)
    )
"""


class Config(gws.Config):
    """Postgres storage provider. (added in 8.4)"""

    dbUid: Optional[str]
    """Database provider uid."""
    tableName: str
    """Table name for the storage."""


class Object(gws.StorageProvider):
    db: provider.Object
    tableName: str

    def configure(self):
        self.configure_provider()
        self.configure_table()

    def configure_table(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        if not self.db.has_table(self.tableName):
            raise gws.ConfigurationError(f'table {self.tableName!r} not found')

    def configure_provider(self):
        return gws.config.util.configure_database_provider_for(self)

    def list_names(self, category):
        with self.db.connect() as conn:
            tab = self._table()
            rs = conn.fetch_all(tab.select().where(tab.c.category == category).with_only_columns(tab.c.name))
            return sorted(rec['name'] for rec in rs)

    def read(self, category, name):
        with self.db.connect() as conn:
            tab = self._table()
            rec = conn.fetch_first(tab.select().where(tab.c.category == category, tab.c.name == name).limit(1))
            if rec:
                return gws.StorageRecord(**rec)

    def write(self, category, name, data, user_uid):
        with self.db.connect() as conn:
            tab = self._table()
            sql = (
                pg_insert(tab)
                .values(category=category, name=name, user_uid=user_uid, data=data)
                .on_conflict_do_update(
                    index_elements=['category', 'name'],
                    set_=dict(
                        user_uid=user_uid,
                        data=data,
                        updated=sa.func.now(),
                    )
                )
            )
            conn.exec_commit(sql)

    def delete(self, category, name):
        with self.db.connect() as conn:
            tab = self._table()
            sql = tab.delete().where(tab.c.category == category, tab.c.name == name)
            conn.exec_commit(sql)

    def _table(self):
        return self.db.table(self.tableName)
