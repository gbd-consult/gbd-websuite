import time
import sqlite3

import gws
import gws.tools.json2

DB_PATH = gws.MISC_DIR + '/storage.sqlite'


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS storage(
        category TEXT,
        name TEXT,
        user_uid TEXT,
        payload TEXT,
        created INTEGER,
        PRIMARY KEY (name, category)
    ) WITHOUT ROWID''')
    conn.row_factory = sqlite3.Row
    return conn


def _ts():
    return int(time.time())


def put(category, name, user_uid, data):
    with _db() as db:
        db.execute('DELETE FROM storage WHERE category=? AND name=?', [category, name])
        db.execute('''INSERT INTO storage(
                category, name, user_uid, payload, created)
                VALUES(?,?,?,?,?)
        ''', [category, name, user_uid, gws.tools.json2.to_string(data), _ts()])


def get(category, name, user_uid):
    with _db() as db:
        for rec in db.execute('SELECT * FROM storage WHERE category=? AND name=? AND user_uid=?', [category, name, user_uid]):
            return gws.tools.json2.from_string(rec['payload'])


def get_names(category, user_uid):
    names = []

    with _db() as db:
        for rec in db.execute('SELECT name FROM storage WHERE category=? AND user_uid=?', [category, user_uid]):
            names.append(rec['name'])

    return sorted(names)
