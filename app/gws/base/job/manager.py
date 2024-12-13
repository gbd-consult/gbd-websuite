"""Job manager."""

from typing import Optional

import gws
import gws.lib.importer
import gws.lib.jsonx
import gws.lib.sqlitex


class Object(gws.JobManager):
    dbPath: str
    table = 'jobs'

    def configure(self):
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/jobs82.sqlite')

    def create_job(self, user, worker, payload=None):
        job_uid = gws.u.random_string(64)
        gws.log.debug(f'JOB {job_uid}: creating: {worker=}  {user.uid=}')

        self._db().insert(self.table, dict(
            uid=job_uid,
            user_uid=user.uid,
            str_user=self.root.app.authMgr.serialize_user(user),
            state=gws.JobState.open,
            worker=worker,
            payload=gws.lib.jsonx.to_string(payload or {}),
            error='',
            created=gws.u.stime(),
            updated=gws.u.stime()
        ))

        return self.get_job(job_uid)

    def get_job(self, job_uid: str) -> Optional[gws.Job]:
        rs = self._db().select(f'SELECT * FROM {self.table} WHERE uid=:uid', uid=job_uid)
        if not rs:
            return

        rec = rs[0]

        user = None
        if rec.get('str_user'):
            user = self.root.app.authMgr.unserialize_user(rec.get('str_user'))

        return gws.Job(
            uid=rec['uid'],
            state=rec['state'],
            error=rec['error'],
            payload=gws.Data(gws.lib.jsonx.from_string(rec['payload'] or '{}')),
            user=user or self.root.app.authMgr.guestUser,
            worker=rec['worker'],
        )

    def get_job_for(self, job_uid: str, user: gws.User) -> Optional[gws.Job]:
        job = self.get_job(job_uid)
        if not job:
            gws.log.error(f'JOB {job_uid}: not found')
            return
        if job.user.uid != user.uid:
            gws.log.error(f'JOB {job_uid}: wrong user (job={job.user.uid!r} user={user.uid!r})')
            return
        return job

    def update_job(self, job: gws.Job, state=None, error=None, payload=None) -> Optional[gws.Job]:
        cur_job = self.get_job(job.uid)
        if not cur_job:
            gws.log.error(f'JOB {job.uid} update: not found')
            return

        params = {}
        if state is not None:
            params['state'] = state
        if error is not None:
            params['error'] = error
        if payload is not None:
            params['payload'] = gws.lib.jsonx.to_string(gws.u.merge(cur_job.payload, payload))
        params['updated'] = gws.u.stime()

        vals = ','.join(f'{k}=:{k}' for k in params)
        self._db().execute(
            f'UPDATE {self.table} SET {vals} WHERE uid=:uid',
            uid=job.uid, **params
        )

        gws.log.debug(f'JOB {job.uid} update: {params=}')
        return self.get_job(job.uid)

    def run_job(self, job: gws.Job):
        job_uid = job.uid
        tmp = gws.u.random_string(64)

        self._db().execute(
            f'UPDATE {self.table} SET state=:tmp WHERE uid=:uid AND state=:state',
            uid=job_uid, tmp=tmp, state=gws.JobState.open
        )

        job = self.get_job(job_uid)
        if not job:
            raise gws.Error(f'JOB {job_uid}: not found')
        if job.state != tmp:
            raise gws.Error(f'JOB {job_uid}: invalid state={job.state!r}')

        self._db().execute(
            f'UPDATE {self.table} SET state=:new_state WHERE uid=:uid',
            uid=job.uid, new_state=gws.JobState.running
        )

        job = self.get_job(job_uid)

        try:
            mod_path, _, fn_name = job.worker.rpartition('.')
            mod = gws.lib.importer.import_from_path(mod_path)
            fn = getattr(mod, fn_name)
            fn(self.root, job)
        except gws.JobTerminated as exc:
            gws.log.error(f'JOB {job_uid}: JobTerminated: {exc.args[0]!r}')
            self.update_job(job, state=gws.JobState.error)
        except Exception as exc:
            gws.log.error(f'JOB {job_uid}: FAILED {exc=}')
            gws.log.exception()
            self.update_job(job, state=gws.JobState.error, error=repr(exc))

        return self.get_job(job_uid)

    def cancel_job(self, job: gws.Job) -> Optional[gws.Job]:
        return self.update_job(job, state=gws.JobState.cancel)

    def remove_job(self, job: gws.Job):
        self._db().execute(f'DELETE FROM {self.table} WHERE uid=:uid', uid=job.uid)

    def job_response(self, job):

        def _progress():
            if job.state == gws.JobState.complete:
                return 100
            if job.state != gws.JobState.running:
                return 0
            step_count = job.payload.get('stepCount', 0)
            if not step_count:
                return 0
            step = job.payload.get('step', 0)
            return int(min(100.0, step * 100.0 / step_count))

        return gws.JobResponse(
            jobUid=job.uid,
            state=job.state,
            progress=_progress(),
            stepName=job.payload.get('stepName', ''),
        )

    ##

    _sqlitex: gws.lib.sqlitex.Object

    def _db(self):
        if getattr(self, '_sqlitex', None) is None:
            ddl = f'''
                CREATE TABLE IF NOT EXISTS {self.table} (
                    uid      TEXT NOT NULL PRIMARY KEY,
                    user_uid TEXT NOT NULL,
                    str_user TEXT NOT NULL,
                    worker   TEXT NOT NULL,
                    state    TEXT NOT NULL,
                    error    TEXT NOT NULL,
                    payload  TEXT NOT NULL,
                    created  INTEGER NOT NULL,
                    updated  INTEGER NOT NULL
                )
            '''
            self._sqlitex = gws.lib.sqlitex.Object(self.dbPath, ddl)
        return self._sqlitex
