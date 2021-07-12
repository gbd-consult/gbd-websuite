import os
import time

import gws
import gws.types as t
import gws.config
import gws.lib.json2
import gws.lib.sqlite

_DEFAULT_PATH = gws.MISC_DIR + '/storage5.sqlite'


class Object(gws.Node):
    def __init__(self):
        super().__init__()
        self.path = ''

    def configure(self):
        
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

    def read(self, entry: gws.StorageEntry) -> gws.StorageRecord:
        rec = None
        with self._connect() as conn:
            rs = conn.execute(
                'SELECT * FROM storage WHERE category=? AND name=? LIMIT 1',
                [entry.category, entry.name])
            for r in rs:
                rec = gws.StorageRecord(**r)
        return rec

    def write(self, entry, user, data: str) -> gws.StorageRecord:
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

    def dir(self, category: str) -> t.List[gws.StorageEntry]:
        ls = []
        with self._connect() as conn:
            rs = conn.execute(
                'SELECT category, name FROM storage WHERE category=? ORDER BY name',
                [category]
            )
            for r in rs:
                ls.append(gws.StorageEntry(**r))
        return ls

    def delete(self, category: str, name: str):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage WHERE category=? AND name=?', [category, name])

    def reset(self):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage')

    def _connect(self):
        return gws.lib.sqlite.connect(self.path)
