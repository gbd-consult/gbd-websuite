import os
import sqlite3
import time

import gws
import gws.config
import gws.tools.json2
import gws.tools.misc

import gws.types as t

_DEFAULT_PATH = gws.MISC_DIR + '/storage5.sqlite'


class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.path = ''

    def configure(self):
        super().configure()
        self.path = self.var('path') or _DEFAULT_PATH
        with self._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS storage(
                category TEXT,
                name TEXT,
                user_fid TEXT,
                data TEXT,
                created INTEGER,
                updated INTEGER,
                PRIMARY KEY (category, name)
            ) WITHOUT ROWID''')
        os.chown(self.path, gws.UID, gws.GID)

    def read(self, entry: t.StorageEntry) -> t.StorageRecord:
        rec = None
        with self._connect() as conn:
            rs = conn.execute(
                'SELECT * FROM storage WHERE category=? AND name=? LIMIT 1',
                [entry.category, entry.name])
            for r in rs:
                rec = t.StorageRecord(**r)
        return rec

    def write(self, entry, user, data: str) -> t.StorageRecord:
        if not entry.category or not entry.name:
            raise ValueError('empty category/name')

        with self._connect() as conn:
            conn.execute('''INSERT OR REPLACE INTO storage(
                category, name, user_fid, data, created)
                VALUES(?,?,?,?,?)
            ''', [
                entry.category,
                entry.name,
                user.fid,
                data,
                int(time.time()),
            ])
        return self.read(entry)

    def dir(self, category: str) -> t.List[t.StorageEntry]:
        ls = []
        with self._connect() as conn:
            rs = conn.execute(
                'SELECT category, name FROM storage WHERE category=? ORDER BY name',
                [category]
            )
            for r in rs:
                ls.append(t.StorageEntry(**r))
        return ls

    def reset(self):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage')

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn
