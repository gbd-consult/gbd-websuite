import importlib

import gws
import gws.config
import gws.tools.json2

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


def create(uid, user: t.IUser, worker: str, project_uid=None, args=None):
    if user:
        fid = user.fid
        str_user = gws.config.root().application.auth.serialize_user(user)
    else:
        fid = str_user = ''
    gws.log.debug('creating job', worker, fid)
    storage.create(uid)
    storage.update(
        uid,
        user_fid=fid,
        str_user=str_user,
        project_uid=project_uid,
        worker=worker,
        args=gws.tools.json2.to_string(args),
        steps=0,
        step=0,
        state=State.open,
    )
    return get(uid)


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
        self.project_uid = ''
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

        self.args = gws.tools.json2.from_string(self.args)
        self.result = gws.tools.json2.from_string(self.result)

    @property
    def user(self) -> t.Optional[t.IUser]:
        if self.str_user:
            return gws.config.root().application.auth.unserialize_user(self.str_user)

    @property
    def progress(self) -> int:
        if self.state == State.complete:
            return 100
        if self.steps and self.state == State.running:
            return min(100, int((self.step or 0) * 100 / self.steps))
        return 0

    def run(self):
        if self.state != State.open:
            gws.log.error(f'job={self.uid!r} invalid state for run={self.state!r}')
            return

        mod_name, _, fn_name = self.worker.rpartition('.')
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)

        root = gws.config.root()

        try:
            fn(root, self)
        except Exception as e:
            gws.log.error('job: FAILED', self.uid)
            self.update(state=State.error, error=repr(e))
            raise

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'result' in kwargs:
            kwargs['result'] = gws.tools.json2.to_string(kwargs['result'])

        storage.update(self.uid, **kwargs)

    def cancel(self):
        self.update(state=State.cancel)

    def remove(self):
        storage.remove(self.uid)
