import importlib

import gws
import gws.lib.jsonx
import gws.types as t

from . import storage


class State(t.Enum):
    init = 'init'
    """the job is being created"""
    open = 'open'
    """the job is just created and waiting for start"""
    running = 'running'
    """the job is running"""
    complete = 'complete'
    """the job has been completed successfully"""
    error = 'error'
    """there was an error"""
    cancel = 'cancel'
    """the job was cancelled"""


class Error(gws.Error):
    pass


class PrematureTermination(Exception):
    pass


def create(root: gws.IRoot, user: gws.IUser, worker: str, payload: dict = None) -> 'Job':
    uid = gws.random_string(64)
    gws.log.debug(f'JOB {uid}: creating: {worker=}  {user.uid=}')
    storage.create(uid)
    storage.update(
        uid,
        user_uid=user.uid,
        str_user=root.app.authMgr.serialize_user(user),
        worker=worker,
        payload=gws.lib.jsonx.to_string(payload or {}),
        state=State.open,
        error='',
    )
    return get(root, uid)


def run(root: gws.IRoot, uid):
    job = get(root, uid)
    if not job:
        raise gws.Error('invalid job_uid {uid!r}')
    job.run()


def get(root: gws.IRoot, uid) -> t.Optional['Job']:
    rec = storage.find(uid)
    if rec:
        return Job(root, rec)


def get_for(root: gws.IRoot, user, uid) -> t.Optional['Job']:
    job = get(root, uid)
    if not job:
        gws.log.error(f'JOB {uid}: not found')
        return
    if job.user.uid != user.uid:
        gws.log.error(f'JOB {uid}: wrong user (job={job.user.uid!r} user={user.uid!r})')
        return
    return job


def remove(uid):
    storage.remove(uid)


class Job:
    error: str
    payload: dict
    state: State
    uid: str
    user: gws.IUser
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
        if self.state != State.open:
            gws.log.error(f'JOB {self.uid}: invalid state for run={self.state!r}')
            return

        # @TODO lock
        self.update(state=State.running)

        try:
            mod_name, _, fn_name = self.worker.rpartition('.')
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            fn(self.root, self)
        except PrematureTermination as exc:
            gws.log.error(f'JOB {self.uid}: PrematureTermination: {exc.args[0]!r}')
            self.update(state=State.error)
        except Exception as exc:
            gws.log.error(f'JOB {self.uid}: FAILED')
            gws.log.exception()
            self.update(state=State.error, error=repr(exc))

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
        self.update(state=State.cancel)

    def remove(self):
        storage.remove(self.uid)
