import importlib

import gws
import gws.lib.json2
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


def create(root: gws.IRoot, uid, user: gws.IUser, worker: str, project_uid=None, args=None):
    if user:
        user_uid = user.uid
        str_user = root.application.auth.serialize_user(user)
    else:
        user_uid = str_user = ''
    gws.log.debug('creating job', worker, user_uid)
    storage.create(uid)
    storage.update(
        uid,
        user_uid=user_uid,
        str_user=str_user,
        project_uid=project_uid,
        worker=worker,
        args=gws.lib.json2.to_string(args),
        steps=0,
        step=0,
        state=State.open,
    )
    return get(root, uid)


def run(root: gws.IRoot, uid):
    job = get(root, uid)
    if not job:
        raise ValueError('invalid job_uid {uid!r}')
    gws.log.debug('running job', job.uid)
    job.run()


def get(root: gws.IRoot, uid):
    rec = storage.find(uid)
    if rec:
        return Job(root, rec)


def remove(uid):
    storage.remove(uid)


def get_for(root: gws.IRoot, user, uid):
    job = get(root, uid)
    if not job:
        gws.log.error(f'job={uid!r}: not found')
        return
    if job.user_uid != user.uid:
        gws.log.error(f'job={uid!r} wrong user (job={job.user_uid!r} user={user.uid!r})')
        return
    return job


class Job:
    def __init__(self, root: gws.IRoot, rec):
        self.root = root

        self.uid = ''
        self.user_uid = ''
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

        self.args = gws.lib.json2.from_string(self.args)
        self.result = gws.lib.json2.from_string(self.result)

    @property
    def user(self) -> gws.IUser:
        auth = self.root.application.auth
        if self.str_user:
            user = auth.unserialize_user(self.str_user)
            if user:
                return user
        return auth.guest_user

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

        try:
            fn(self.root, self)
        except Exception as e:
            gws.log.error('job: FAILED', self.uid)
            self.update(state=State.error, error=repr(e))
            raise

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'result' in kwargs:
            kwargs['result'] = gws.lib.json2.to_string(kwargs['result'])

        storage.update(self.uid, **kwargs)

    def cancel(self):
        self.update(state=State.cancel)

    def remove(self):
        storage.remove(self.uid)
