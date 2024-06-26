from typing import Optional

import importlib

import gws
import gws.lib.jsonx
import gws.lib.sqlitex


class Error(gws.Error):
    pass


class PrematureTermination(Exception):
    pass


_DB_PATH = gws.c.PRINT_DIR + '/jobs81.sqlite'

_TABLE = 'job'

_INIT_DDL = f'''
    CREATE TABLE IF NOT EXISTS {_TABLE} (
        uid      TEXT NOT NULL PRIMARY KEY,
        user_uid TEXT NOT NULL,
        str_user TEXT NOT NULL,
        worker   TEXT NOT NULL,
        payload  TEXT NOT NULL,
        state    TEXT NOT NULL,
        error    TEXT NOT NULL,
        created  INTEGER NOT NULL,
        updated  INTEGER NOT NULL
    )            
'''


def create(root: gws.Root, user: gws.User, worker: str, payload: dict = None) -> 'Object':
    uid = gws.u.random_string(64)
    gws.log.debug(f'JOB {uid}: creating: {worker=}  {user.uid=}')

    _db().insert(_TABLE, dict(
        uid=uid,
        user_uid=user.uid,
        str_user=root.app.authMgr.serialize_user(user),
        worker=worker,
        payload=gws.lib.jsonx.to_string(payload or {}),
        state=gws.JobState.open,
        error='',
        created=gws.u.stime(),
        updated=gws.u.stime()
    ))

    job = get(root, uid)
    if not job:
        raise gws.Error(f'error creating job {uid=}')
    return job


def run(root: gws.Root, uid):
    job = get(root, uid)
    if not job:
        raise gws.Error(f'invalid job {uid=}')
    job.run()


def get(root: gws.Root, uid) -> Optional['Object']:
    rs = _db().select(f'SELECT * FROM {_TABLE} WHERE uid=:uid', uid=uid)
    if rs:
        return Object(root, rs[0])


def remove(uid):
    _db().execute(f'DELETE FROM {_TABLE} WHERE uid=:uid', uid=uid)


##

class Object(gws.Job):
    worker: str

    def __init__(self, root: gws.Root, rec):
        self.root = root

        self.error = rec['error']
        self.payload = gws.lib.jsonx.from_string(rec['payload'])
        self.state = rec['state']
        self.uid = rec['uid']
        self.user = self._get_user(rec)
        self.worker = rec['worker']

    def _get_user(self, rec) -> gws.User:
        auth = self.root.app.authMgr
        if rec.get('str_user'):
            user = auth.unserialize_user(rec.get('str_user'))
            if user:
                return user
        return auth.guestUser

    def run(self):
        if self.state != gws.JobState.open:
            gws.log.error(f'JOB {self.uid}: invalid state for run={self.state!r}')
            return

        # @TODO lock
        self.update(state=gws.JobState.running)

        try:
            mod_name, _, fn_name = self.worker.rpartition('.')
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            fn(self.root, self)
        except PrematureTermination as exc:
            gws.log.error(f'JOB {self.uid}: PrematureTermination: {exc.args[0]!r}')
            self.update(state=gws.JobState.error)
        except Exception as exc:
            gws.log.error(f'JOB {self.uid}: FAILED')
            gws.log.exception()
            self.update(state=gws.JobState.error, error=repr(exc))

    def update(self, payload=None, state=None, error=None):
        rec = {
            'updated': gws.u.stime(),
        }

        if payload is not None:
            rec['payload'] = gws.lib.jsonx.to_string(payload)
        if state:
            rec['state'] = state
        if error:
            rec['error'] = error

        vals = ','.join(f'{k}=:{k}' for k in rec)

        _db().execute(f'UPDATE {_TABLE} SET {vals} WHERE uid=:uid', uid=self.uid, **rec)

        gws.log.debug(f'JOB {self.uid}: update: {rec=}')

    def cancel(self):
        self.update(state=gws.JobState.cancel)

    def remove(self):
        _db().execute(f'DELETE FROM {_TABLE} WHERE uid=:uid', uid=self.uid)


##


_sqlitex: Optional[gws.lib.sqlitex.Object] = None


def _db() -> gws.lib.sqlitex.Object:
    global _sqlitex

    if _sqlitex is None:
        _sqlitex = gws.lib.sqlitex.Object(_DB_PATH, _INIT_DDL)
    return _sqlitex
