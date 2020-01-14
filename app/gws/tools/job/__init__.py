import importlib

import gws
import gws.common.auth

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


def create(uid, user: t.IUser, worker: str, args=None):
    if user:
        fid = user.fid
        str_user = gws.common.auth.serialize_user(user)
    else:
        fid = str_user = ''
    gws.log.debug('creating job', worker, fid)
    storage.create(uid)
    storage.update(
        uid,
        user_fid=fid,
        str_user=str_user,
        worker=worker,
        args=args or '',
        steps=0,
        step=0,
        state=State.open,
        created=storage.timestamp(),
    )
    return get(uid)


def url(uid):
    return gws.SERVER_ENDPOINT + f'?cmd=assetHttpGetResult&jobUid={uid}'


def get(uid):
    rec = storage.find(uid)
    if rec:
        return Job(rec)


def remove(uid):
    storage.remove(uid)


def get_for(user, uid):
    job = get(uid)
    if not job:
        gws.log.error(f'job={uid!r}: not found')
        return
    if job.user_fid != user.fid:
        gws.log.error(f'job={uid!r} wrong user (job={job.user_fid!r} user={user.fid!r})')
        return
    return job


class Job:
    def __init__(self, rec):
        self.uid = ''
        self.user_fid = ''
        self.str_user = ''
        self.worker = ''
        self.args = ''
        self.steps = 0
        self.step = 0
        self.state = ''
        self.steptype = ''
        self.stepname = ''
        self.error = ''
        self.result = ''
        self.created = 0
        self.updated = 0

        for k, v in rec.items():
            setattr(self, k, v)

    @property
    def user(self) -> t.Optional[t.IUser]:
        if self.str_user:
            return gws.common.auth.unserialize_user(self.str_user)

    def run(self):
        if self.state != State.open:
            gws.log.error(f'job={self.uid!r} invalid state for run={self.state!r}')
            return

        mod_name, _, fn_name = self.worker.rpartition('.')
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)

        try:
            fn(self)
        except Exception as e:
            gws.log.error('job: FAILED', self.uid)
            self.update(State.error, error=repr(e))
            raise

    def update(self, state, **kwargs):
        storage.update(self.uid, state=state, **kwargs)

    def cancel(self):
        self.update(State.cancel)

    def remove(self):
        storage.remove(self.uid)
