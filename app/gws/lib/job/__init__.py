import importlib

import gws
import gws.lib.jsonx
import gws.types as t

from . import storage


class Error(gws.Error):
    pass


class PrematureTermination(Exception):
    pass


def create(root: gws.IRoot, user: gws.IUser, worker: str, payload: dict = None) -> 'Object':
    uid = gws.random_string(64)
    gws.log.debug(f'JOB {uid}: creating: {worker=}  {user.uid=}')
    storage.create(uid)
    storage.update(
        uid,
        user_uid=user.uid,
        str_user=root.app.authMgr.serialize_user(user),
        worker=worker,
        payload=gws.lib.jsonx.to_string(payload or {}),
        state=gws.JobState.open,
        error='',
    )
    return get(root, uid)


def run(root: gws.IRoot, uid):
    job = get(root, uid)
    if not job:
        raise gws.Error('invalid job_uid {uid!r}')
    job.run()


def get(root: gws.IRoot, uid) -> t.Optional['Object']:
    rec = storage.find(uid)
    if rec:
        return Object(root, rec)


def remove(uid):
    storage.remove(uid)


class Object:
    worker: str

    def __init__(self, root: gws.IRoot, rec):
        self.root = root

        self.error = rec['error']
        self.payload = gws.lib.jsonx.from_string(rec['payload'])
        self.state = rec['state']
        self.uid = rec['uid']
        self.user = self._get_user(rec)
        self.worker = rec['worker']

    def _get_user(self, rec) -> gws.IUser:
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
        rec = {}

        if payload is not None:
            rec['payload'] = gws.lib.jsonx.to_string(payload)
        if state:
            rec['state'] = state
        if error:
            rec['error'] = error

        storage.update(self.uid, **rec)
        gws.log.debug(f'JOB {self.uid}: update: {rec=}')

    def cancel(self):
        self.update(state=gws.JobState.cancel)

    def remove(self):
        storage.remove(self.uid)
