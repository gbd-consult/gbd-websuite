import time

import gws
import gws.lib.sqlite

_DB_PATH = gws.c.PRINT_DIR + '/jobs8.sqlite'


def _db():
    return gws.lib.sqlite.connect(_DB_PATH)


def _ensure_table(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS jobs(
        uid TEXT PRIMARY KEY,
        user_uid TEXT,
        str_user TEXT,
        worker TEXT,
        payload TEXT,
        state TEXT,
        error TEXT DEFAULT '',
        created INTEGER,
        updated INTEGER
    ) WITHOUT ROWID''')


def timestamp():
    return int(time.time())


def find(uid):
    with _db() as db:
        for rec in db.execute('SELECT * FROM jobs WHERE uid=?', [uid]):
            return dict(rec)


def create(uid):
    with _db() as db:
        _ensure_table(db)
        db.execute('INSERT INTO jobs(uid,created) VALUES (?,?)', [uid, timestamp()])
    return uid


def update(uid, **kwargs):
    with _db() as db:
        sql = ['updated=?']
        params = [timestamp()]
        for k, v in kwargs.items():
            sql.append(k + '=?')
            params.append(v)
        params.append(uid)
        sql = 'UPDATE jobs SET %s WHERE uid=?' % ','.join(sql)
        db.execute(sql, params)


def remove(uid):
    with _db() as db:
        sql = 'DELETE FROM jobs WHERE uid=?'
        db.execute(sql, [uid])
