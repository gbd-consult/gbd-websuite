import sqlite3
import time

import gws
import gws.config
import gws.tools.json2
import gws.tools.misc
import gws.types as t

_DEFAULT_PATH = gws.MISC_DIR + '/store.sqlite'


class Config(t.WithType):
    """Sqlite-based storage"""

    path: t.Optional[str]


class Object(gws.Object, t.StorageInterface):
    def configure(self):
        super().configure()
        self.path = self.var('path') or _DEFAULT_PATH
        with self._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS storage(
                category TEXT,
                name TEXT,
                user TEXT,
                roles TEXT,
                data TEXT,
                created INTEGER,
                PRIMARY KEY (category, name)
            ) WITHOUT ROWID''')

    def read(self, entry, user):
        with self._connect() as conn:
            r = self._read(conn, entry)
            if self.can_read(r, user):
                return gws.tools.json2.from_string(r['data'])

    def write(self, entry, user, data):
        if not entry.category or not entry.name:
            raise ValueError('empty category/name')

        with self._connect() as conn:
            r = self._read(conn, entry)
            if r and not self.can_write(r, user):
                gws.log.info(f"storage.write: denied r={r['category']!r}:{r['name']!r} user={user.full_uid}")
                return False

            conn.execute('''INSERT OR REPLACE INTO storage(
                category, name, user, roles, data, created)
                VALUES(?,?,?,?,?,?)
            ''', [
                entry.category,
                entry.name,
                user.full_uid,
                ','.join(user.roles),
                gws.tools.json2.to_string(data),
                int(time.time())
            ])
        return True

    def dir(self, user):
        entries = []
        with self._connect() as conn:
            rs = conn.execute('SELECT * FROM storage ORDER BY category, name')
            for r in rs:
                if self.can_read(r, user):
                    entries.append(t.StorageEntry({
                        'category': r['category'],
                        'name': r['name']
                    }))
        return entries

    def can_read(self, r, user):
        return r['user'] == user.full_uid

    def can_write(self, r, user):
        return r['user'] == user.full_uid

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _read(self, conn, entry):
        rs = conn.execute(
            'SELECT * FROM storage WHERE category=? AND name=?',
            [entry.category, entry.name])
        for r in rs:
            return r
