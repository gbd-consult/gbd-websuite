import importlib

import gws
import gws.lib.jsonx
import gws.types as t

from . import storage


class State(t.Enum):
    init = 'init'  #: the job is being created
    open = 'open'  #: the job is just created and waiting for start
    running = 'running'  #: the job is running
    complete = 'complete'  #: the job has been completed successfully
    error = 'error'  #: there was an error
    cancel = 'cancel'  #: the job was cancelled


class Error(gws.Error):
    pass


class PrematureTermination(Exception):
    pass


def create(root: gws.IRoot, uid, user: gws.IUser, worker: str, payload=None) -> 'Job':
    if user:
        user_uid = user.uid
        str_user = root.app.auth.serialize_user(user)
    else:
        user_uid = str_user = ''
    gws.log.debug('creating job', worker, user_uid)
    storage.create(uid)
    storage.update(
        uid,
        user_uid=user_uid,
        str_user=str_user,
        worker=worker,
        payload=gws.lib.jsonx.to_string(payload),
        state=State.open,
        error='',
    )
    return get(root, uid)


def run(root: gws.IRoot, uid):
    job = get(root, uid)
    if not job:
        raise gws.Error('invalid job_uid {uid!r}')
    gws.log.debug('running job', job.uid)
    job.run()


def get(root: gws.IRoot, uid) -> t.Optional['Job']:
    rec = storage.find(uid)
    if rec:
        return Job(root, rec)


def remove(uid):
    storage.remove(uid)


def get_for(root: gws.IRoot, user, uid) -> t.Optional['Job']:
    job = get(root, uid)
    if not job:
        gws.log.error(f'job={uid!r}: not found')
        return
    if job.user.uid != user.uid:
        gws.log.error(f'job={uid!r} wrong user (job={job.user.uid!r} user={user.uid!r})')
        return
    return job


class Job:
    uid: str
    user: gws.IUser
    state: State
    error: str
    payload: gws.Data
    worker: str

    def __init__(self, root: gws.IRoot, rec):
        self.root = root

        self.uid = rec['uid']
        self.user = self._get_user(rec)
        self.worker = rec['worker']
        self.payload = gws.Data(gws.lib.jsonx.from_string(rec.get('payload', '')))
        self.state = rec['state']
        self.error = rec['error']

    def _get_user(self, rec) -> gws.IUser:
        auth = self.root.app.auth
        if rec.get('str_user'):
            user = auth.unserialize_user(rec.get('str_user'))
            if user:
                return user
        return auth.guestUser

    def run(self):
        if self.state != State.open:
            gws.log.error(f'job={self.uid!r} invalid state for run={self.state!r}')
            return

        mod_name, _, fn_name = self.worker.rpartition('.')
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)

        try:
            fn(self.root, self)
        except Exception as e:
            gws.log.error('job: FAILED', self.uid)
            self.update(state=State.error, error=repr(e))
            raise

    def update(self, payload=None, state=None, error=None):
        rec = {}

        if payload is not None:
            rec['payload'] = gws.lib.jsonx.to_string(payload)
        if state:
            rec['state'] = state
        if error:
            rec['error'] = error

        storage.update(self.uid, **rec)

    def cancel(self):
        self.update(state=State.cancel)

    def remove(self):
        storage.remove(self.uid)
