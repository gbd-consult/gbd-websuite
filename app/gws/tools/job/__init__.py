import importlib
import gws
import gws.config
import gws.auth.api
import gws.tools.json2 as json2
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


# @TODO: merge with sessions serializer

# def _encode(a):
#     if isinstance(a, dict):
#         return {k: _encode(v) for k, v in a.items()}
#     if isinstance(a, list):
#         return [_encode(v) for v in a]
#     if isinstance(a, gws.auth.api.User):
#         prov = a.provider
#         return {
#             '$user': True,
#             'provider_uid': prov.uid,
#             'user_uid': a.uid,
#             'str_user': prov.marshal_user(a),
#         }
#     if isinstance(a, gws.Object):
#         return {'$object': [a.klass, a.uid]}
#     if isinstance(a, t.Data):
#         return {'$data': [a.__class__.__name__, _encode(vars(a))]}
#     return a
#
#
# def _decode(a):
#     if isinstance(a, dict):
#         if '$user' in a:
#             prov: t.AuthProviderInterface = gws.config.find('gws.ext.auth.provider', a['provider_uid'])
#             return prov.unmarshal_user(a['user_uid'], a['str_user'])
#         if '$object' in a:
#             return gws.config.find(*a['$object'])
#         if '$data' in a:
#             cls, v = a['$data']
#             # since all Data classes are just dicts, it's safe to use Data here
#             # instead of the specific class (really?)
#             return t.Data(_decode(v))
#         return {k: _decode(v) for k, v in a.items()}
#
#     if isinstance(a, list):
#         return [_decode(v) for v in a]
#
#     return a


def create(uid, user_uid, worker, args=None):
    gws.log.debug('creating job', worker, user_uid)
    storage.create(uid)
    storage.update(
        uid,
        user_uid=user_uid,
        worker=worker,
        args=args or '',
        steps=0,
        step=0,
        state=State.open,
        created=storage.timestamp(),
    )
    return get(uid)


def get(uid):
    rec = storage.find(uid)
    if rec:
        return Job(rec)


def get_for(user, uid):
    job = get(uid)
    if not job:
        gws.log.error(f'job={uid!r}: not found')
        return
    if job.user_uid != user.full_uid:
        gws.log.error(f'job={uid!r} wrong user (job={job.user_uid!r} user={user.full_uid!r})')
        return
    return job


class Job:
    def __init__(self, rec):
        self.uid = rec['uid']
        self.user_uid = rec['user_uid']
        self.worker = rec['worker']
        self.args = rec['args']
        self.steps = rec['steps']
        self.step = rec['step']
        self.state = rec['state']
        self.otype = rec['otype']
        self.oname = rec['oname']
        self.error = rec['error']
        self.result = rec['result']
        self.created = rec['created']
        self.updated = rec['updated']

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
