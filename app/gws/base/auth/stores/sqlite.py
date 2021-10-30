import gws
import gws.lib.date
import gws.lib.os2
import gws.lib.sqlite


class SessionStore:
    db_path: str

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init()

    def init(self):
        with self._connection as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS sess(
                uid TEXT PRIMARY KEY,
                method_uid TEXT,
                typ TEXT,
                provider_uid TEXT,
                user_uid TEXT,
                str_user TEXT,
                str_data TEXT,
                created INTEGER,
                updated INTEGER
            ) WITHOUT ROWID''')

    @property
    def _connection(self):
        return gws.lib.sqlite.connect(self.db_path)

    def count(self):
        with self._connection as conn:
            return conn.execute('SELECT COUNT(*) FROM sess').fetchone(self)[0]

    def cleanup(self, lifetime):
        with self._connection as conn:
            conn.execute('DELETE FROM sess WHERE updated < ?', [gws.lib.date.timestamp() - lifetime])

    def find(self, uid):
        with self._connection as conn:
            for r in conn.execute('SELECT * FROM sess WHERE uid=?', [uid]):
                return dict(r)

    def create(self, method_uid, typ, provider_uid, user_uid, str_user, str_data=''):
        uid = gws.random_string(64)

        with self._connection as conn:
            conn.execute('''INSERT
                INTO sess(
                    uid,
                    method_uid,
                    typ,
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
                method_uid,
                typ,
                provider_uid,
                user_uid,
                str_user,
                str_data,
                gws.lib.date.timestamp(),
                gws.lib.date.timestamp()
            ])

        gws.log.debug('session: created:', uid)
        return uid

    def update(self, uid, str_data):
        with self._connection as conn:
            conn.execute(
                'UPDATE sess SET str_data=?, updated=? WHERE uid=?',
                [str_data, gws.lib.date.timestamp(), uid])

    def touch(self, uid):
        with self._connection as conn:
            conn.execute(
                'UPDATE sess SET updated=? WHERE uid=?',
                [gws.lib.date.timestamp(), uid])

    def delete(self, uid):
        with self._connection as conn:
            conn.execute('DELETE FROM sess WHERE uid = ?', [uid])

    def delete_all(self):
        gws.lib.os2.unlink(self.db_path)
        self.init()

    def get_all(self):
        rs = []
        with self._connection as conn:
            for r in conn.execute('SELECT * FROM sess ORDER BY updated'):
                rs.append(dict(r))
        return rs
