import os
import time

import gws
import gws.lib.sqlite
import gws.types as t
from .. import types

_DEFAULT_PATH = gws.MISC_DIR + '/storage8.sqlite'


class Object(gws.Object):
    path: str

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

    def read(self, category: str, name: str) -> t.Optional[types.Record]:
        with self._connect() as conn:
            rs = conn.execute(
                'SELECT * FROM storage WHERE category=? AND name=? LIMIT 1',
                [category, name])
            for r in rs:
                return types.Record(**r)

    def write(self, category: str, name: str, data: str, user_fid: str):
        with self._connect() as conn:
            conn.execute('''INSERT OR REPLACE INTO storage(
                category, name, user_fid, data, created)
                VALUES(?,?,?,?,?)
            ''', [
                category,
                name,
                user_fid,
                data,
                int(time.time()),
            ])

    def list(self, category: str) -> t.List[types.Record]:
        with self._connect() as conn:
            rs = conn.execute(
                '''
                    SELECT category,name,user_fid, NULL as data, created, updated 
                    FROM storage WHERE category=? ORDER BY name
                ''',
                [category]
            )
            return [types.Record(**r) for r in rs]

    def delete(self, category: str, name: str):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage WHERE category=? AND name=?', [category, name])

    def reset(self):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage')

    def _connect(self):
        return gws.lib.sqlite.connect(self.path)
