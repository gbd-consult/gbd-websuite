import os
import time

import gws
import gws.lib.sqlite
import gws.types as t

from .. import types

# NB, for compatibility, the DB is 'storage5' and the user column is called 'user_fid'

_DEFAULT_PATH = gws.MISC_DIR + '/storage5.sqlite'


class Object(gws.Node):
    path: str

    def configure(self):
        self.path = self.cfg('path') or _DEFAULT_PATH
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
                return self._record(r)

    def write(self, category: str, name: str, data: str, user_uid: str):
        with self._connect() as conn:
            conn.execute('''INSERT OR REPLACE INTO storage(
                category, name, user_fid, data, created)
                VALUES(?,?,?,?,?)
            ''', [
                category,
                name,
                user_uid,
                data,
                int(time.time()),
            ])

    def list(self, category: str) -> list[types.Record]:
        with self._connect() as conn:
            rs = conn.execute(
                '''
                    SELECT category,name,user_fid, NULL as data, created, updated
                    FROM storage WHERE category=? ORDER BY name
                ''',
                [category]
            )
            return [self._record(r) for r in rs]

    def delete(self, category: str, name: str):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage WHERE category=? AND name=?', [category, name])

    def reset(self):
        with self._connect() as conn:
            conn.execute('DELETE FROM storage')

    def _connect(self):
        return gws.lib.sqlite.connect(self.path)

    def _record(self, r):
        r = dict(r)
        if 'user_fid' in r:
            r['user_uid'] = r.pop('user_fid')
        return types.Record(r)
