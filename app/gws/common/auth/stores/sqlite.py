import os

import gws
import gws.tools.os2
import gws.tools.sqlite
import gws.tools.date

DB_PATH = gws.MISC_DIR + '/sessions6.sqlite'


class SessionStore:
    @property
    def _connection(self):
        return gws.tools.sqlite.connect(DB_PATH)

    def init(self):
        with self._connection as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS sess(
                uid TEXT PRIMARY KEY,
                method_type TEXT,
                session_type TEXT,
                provider_uid TEXT,
                user_uid TEXT,
                str_user TEXT,
                str_data TEXT,
                created INTEGER,
                updated INTEGER
            ) WITHOUT ROWID''')

    def count(self):
        with self._connection as conn:
            return conn.execute('SELECT COUNT(*) FROM sess').fetchone(self)[0]

    def cleanup(self, lifetime):
        with self._connection as conn:
            conn.execute('DELETE FROM sess WHERE updated < ?', [gws.tools.date.timestamp() - lifetime])

    def find(self, uid):
        with self._connection as conn:
            for r in conn.execute('SELECT * FROM sess WHERE uid=?', [uid]):
                return dict(r)

    def create(self, method_type, session_type, provider_uid, user_uid, str_user, str_data=''):
        uid = gws.random_string(64)

        with self._connection as conn:
            conn.execute('''INSERT 
                INTO sess(
                    uid,
                    method_type,
                    session_type,
                    provider_uid,
                    user_uid,
                    str_user,
                    str_data,
                    created,
                    updated
                ) 
                VALUES(?,?,?,?,?,?,?,?,?)
            ''', [
                uid,
                method_type,
                session_type,
                provider_uid,
                user_uid,
                str_user,
                str_data,
                gws.tools.date.timestamp(),
                gws.tools.date.timestamp()
            ])

        gws.log.debug('session: created:', uid)
        return uid

    def update(self, uid, str_data):
        with self._connection as conn:
            conn.execute(
                'UPDATE sess SET str_data=?, updated=? WHERE uid=?',
                [str_data, gws.tools.date.timestamp(), uid])

    def touch(self, uid):
        with self._connection as conn:
            conn.execute(
                'UPDATE sess SET updated=? WHERE uid=?',
                [gws.tools.date.timestamp(), uid])

    def delete(self, uid):
        with self._connection as conn:
            conn.execute('DELETE FROM sess WHERE uid = ?', [uid])

    def delete_all(self):
        gws.tools.os2.unlink(DB_PATH)
        self.init()

    def get_all(self):
        rs = []
        with self._connection as conn:
            for r in conn.execute('SELECT * FROM sess ORDER BY updated'):
                rs.append(dict(r))
        return rs
