"""Job manager."""

from typing import Optional

import gws
import gws.lib.importer
import gws.lib.jsonx
import gws.lib.sqlitex
import gws.lib.mime
import gws.server.spool


class Object(gws.JobManager):
    TABLE = 'jobs'
    DDL = f'''
        CREATE TABLE IF NOT EXISTS {TABLE} (
            uid           TEXT NOT NULL PRIMARY KEY,
            userUid       TEXT DEFAULT '',
            userStr       TEXT DEFAULT '',
            worker        TEXT DEFAULT '',
            state         TEXT DEFAULT '',
            error         TEXT DEFAULT '',
            numSteps      INTEGER DEFAULT 0,
            step          INTEGER DEFAULT 0,
            stepName      TEXT DEFAULT '',
            payload       TEXT DEFAULT '',
            created       INTEGER DEFAULT 0,
            updated       INTEGER DEFAULT 0
        )
    '''

    dbPath: str

    def configure(self):
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/jobs82.sqlite')
        self.root.app.monitor.register_periodic_task(self)

    def periodic_task(self):
        gws.log.info(f'JOB: cleanup')

    def create_job(self, user, worker, payload=None):
        job_uid = gws.u.random_string(64)
        gws.log.debug(f'JOB {job_uid}: creating: {worker=}  {user.uid=}')

        self._db().insert(self.TABLE, dict(uid=job_uid))

        self._write(job_uid, dict(
            uid=job_uid,
            userUid=user.uid,
            userStr=self.root.app.authMgr.serialize_user(user),
            worker=worker,
            state=gws.JobState.open,
            payload=payload,
            created=gws.u.stime(),
        ))

        return self.get_job(job_uid)

    def get_job(self, job_uid: str, user=None, state=None):
        rs = self._db().select(f'SELECT * FROM {self.TABLE} WHERE uid=:uid', uid=job_uid)
        if not rs:
            gws.log.error(f'JOB {job_uid}: not found')
            return

        rec = rs[0]

        job_user = None
        if rec.get('userStr'):
            job_user = self.root.app.authMgr.unserialize_user(rec.get('userStr'))
        if not job_user:
            job_user = self.root.app.authMgr.guestUser

        if user and job_user.uid != user.uid:
            gws.log.error(f'JOB {job_uid}: wrong user {job_user.uid=} {user.uid=}')
            return

        if state and rec.get('state') != state:
            gws.log.error(f'JOB {job_uid}: wrong state {rec.get("state")=} {state=}')

        return gws.Job(
            uid=rec['uid'],
            user=job_user,
            worker=rec['worker'],
            state=rec['state'],
            error=rec['error'],
            numSteps=rec['numSteps'] or 0,
            step=rec['step'] or 0,
            stepName=rec['stepName'] or '',
            payload=gws.lib.jsonx.from_string(rec['payload'] or '{}'),
        )

    def save_job(self, job, **kwargs):
        job = self.get_job(job.uid)
        if not job:
            return

        d = dict(
            state=job.state,
            error=job.error,
            numSteps=job.numSteps,
            step=job.step,
            stepName=job.stepName,
            payload=job.payload,
        )
        d.update(kwargs)

        self._write(job.uid, d)

        return self.get_job(job.uid)

    def _write(self, job_uid, rec):
        rec['updated'] = gws.u.stime()
        rec['payload'] = gws.lib.jsonx.to_string(rec.get('payload') or {})
        gws.log.debug(f'JOB save: {job_uid}: {rec}')
        self._db().update(self.TABLE, rec, job_uid)

    def schedule_job(self, job: gws.Job):
        if gws.server.spool.is_active():
            gws.server.spool.add(job)
            return job
        return self.run_job(job)

    def run_job(self, job: gws.Job):
        job_uid = job.uid
        tmp = gws.u.random_string(64)

        self._db().execute(
            f'UPDATE {self.TABLE} SET state=:tmp WHERE uid=:uid AND state=:state',
            uid=job_uid, tmp=tmp, state=gws.JobState.open
        )

        job = self.get_job(job_uid)
        if job.state != tmp:
            raise gws.Error(f'JOB {job_uid}: invalid state={job.state!r}')

        self._db().execute(
            f'UPDATE {self.TABLE} SET state=:s WHERE uid=:uid',
            uid=job.uid, s=gws.JobState.running
        )

        job = self.get_job(job_uid)

        try:
            mod_path, _, fn_name = job.worker.rpartition('.')
            mod = gws.lib.importer.import_from_path(mod_path)
            worker_fn = getattr(mod, fn_name)
            worker_fn(self.root, job)
        except gws.JobTerminated as exc:
            gws.log.error(f'JOB {job_uid}: JobTerminated: {exc.args[0]!r}')
            job.state = gws.JobState.error
            self.save_job(job)
        except Exception as exc:
            gws.log.error(f'JOB {job_uid}: FAILED {exc=}')
            gws.log.exception()
            job.state = gws.JobState.error
            job.error = repr(exc)
            self.save_job(job)

        return self.get_job(job_uid)

    def cancel_job(self, job: gws.Job) -> Optional[gws.Job]:
        job.state = gws.JobState.cancel
        return self.save_job(job)

    def remove_job(self, job: gws.Job):
        self._db().delete(self.TABLE, job.uid)

    def require_job(self, req, p):
        job = self.root.app.jobMgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'JOB {p.jobUid}: not found')
        return job

    def handle_status_request(self, req, p):
        job = self.require_job(req, p)
        return self.job_status_response(job)

    def handle_cancel_request(self, req, p):
        job = self.require_job(req, p)
        job = self.cancel_job(job)
        return self.job_status_response(job)

    def job_status_response(self, job, **kwargs):
        d = dict(
            jobUid=job.uid,
            state=job.state,
            stepName=job.stepName or '',
            progress=self._get_progress(job),
            output={}
        )
        d.update(kwargs)
        return gws.JobStatusResponse(d)

    def _get_progress(self, job):
        if job.state == gws.JobState.complete:
            return 100
        if job.state != gws.JobState.running:
            return 0
        if not job.numSteps:
            return 0
        return int(min(100.0, job.step * 100.0 / job.numSteps))

    ##

    _sqlitex: gws.lib.sqlitex.Object

    def _db(self):
        if getattr(self, '_sqlitex', None) is None:
            self._sqlitex = gws.lib.sqlitex.Object(self.dbPath, self.DDL)
        return self._sqlitex
