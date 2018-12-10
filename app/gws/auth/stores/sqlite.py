import time
import sqlite3

import gws
import gws.tools.shell

DB_PATH = gws.MISC_DIR + '/sessions.sqlite'


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ts():
    return int(time.time())


def init():
    with _db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS sess(
            uid TEXT PRIMARY KEY,
            provider_uid TEXT,
            user_uid TEXT,
            str_user TEXT,
            str_data TEXT,
            created INTEGER,
            updated INTEGER
        ) WITHOUT ROWID''')


def delete_all():
    with _db() as db:
        db.execute('DROP TABLE sess')


def count():
    with _db() as db:
        return db.execute('SELECT COUNT(*) FROM sess').fetchone()


def cleanup(lifetime):
    with _db() as db:
        db.execute('DELETE FROM sess WHERE updated < ?', [_ts() - lifetime])


def find(uid):
    with _db() as db:
        for r in db.execute('SELECT * FROM sess WHERE uid=?', [uid]):
            return dict(r)


def create(provider_uid, user_uid, str_user, str_data=''):
    uid = gws.random_string(64)
    ts = _ts()

    with _db() as db:
        db.execute('''INSERT 
            INTO sess(
                uid,
                provider_uid,
                user_uid,
                str_user,
                str_data,
                created,
                updated
            ) 
            VALUES(?,?,?,?,?,?,?)
        ''', [
            uid,
            provider_uid,
            user_uid,
            str_user,
            str_data,
            ts,
            ts
        ])

    gws.log.info('session: created:', uid)
    return uid


def update(uid, str_data):
    with _db() as db:
        db.execute('UPDATE sess SET str_data=?, updated=? WHERE uid=?', [str_data, _ts(), uid])


def touch(uid):
    with _db() as db:
        db.execute('UPDATE sess SET updated=? WHERE uid=?', [_ts(), uid])


def delete(uid):
    with _db() as db:
        db.execute('DELETE FROM sess WHERE uid = ?', [uid])


def drop():
    gws.tools.shell.unlink(DB_PATH)
    init()
