"""PostgreSQL storage provider."""

from typing import Optional

import gws
import gws.config.util
import gws.lib.sa as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from . import provider


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
"""DDL of the storage table; ``{table_name}`` is the table name. The provider does not create the table."""


@gws.ext.config.storageProvider('postgres')
class Config(gws.Config):
    """Storage provider that keeps saved user data in a PostgreSQL table."""

    dbUid: Optional[str]
    """UID of the database provider."""
    tableName: str
    """Table for the stored records."""


@gws.ext.object.storageProvider('postgres')
class Object(gws.StorageProvider):
    """Storage provider that keeps records in a PostgreSQL table."""

    db: provider.Object
    """Database provider."""
    tableName: str
    """Table for the stored records."""

    def configure(self):
        self.configure_provider()
        self.configure_table()

    def configure_table(self):
        """Set the table name from the configuration and check that the table exists.

        The table must have the columns given in ``TABLE_DDL``.

        Raises:
            ``gws.ConfigurationError``: If the table does not exist.
        """
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        if not self.db.has_table(self.tableName):
            raise gws.ConfigurationError(f'table {self.tableName!r} not found')

    def configure_provider(self):
        """Set the database provider from ``dbUid``, or the first ``postgres`` provider.

        Returns:
            ``True`` if a provider was set.

        Raises:
            ``gws.Error``: If no provider is found.
        """
        return gws.config.util.configure_database_provider_for(self)

    def list_names(self, category):
        with self.db.begin() as conn:
            tab = self._table()
            rs = conn.fetch_all(tab.select().where(tab.c.category == category).with_only_columns(tab.c.name))
            return sorted(rec['name'] for rec in rs)

    def read(self, category, name):
        with self.db.begin() as conn:
            tab = self._table()
            rec = conn.fetch_first(tab.select().where(tab.c.category == category, tab.c.name == name).limit(1))
            if rec:
                return gws.StorageRecord(**rec)

    def write(self, category, name, data, user_uid):
        with self.db.begin() as conn:
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
            conn.execute(sql)

    def delete(self, category, name):
        with self.db.begin() as conn:
            tab = self._table()
            sql = tab.delete().where(tab.c.category == category, tab.c.name == name)
            conn.execute(sql)

    def _table(self):
        """Return the SQLAlchemy table object of the storage table."""
        return self.db.table(self.tableName)
