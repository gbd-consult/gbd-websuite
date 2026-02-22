"""Job manager."""

from typing import Optional
import sys

import gws
import gws.lib.importer
import gws.lib.jsonx
import gws.lib.sqlitex
import gws.lib.datetimex as dtx
import gws.server.spool


class Object(gws.JobManager):
    TABLE = 'jobs'
    DDL = f"""
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
    """

    dbPath: str

    def configure(self):
        ver = self.root.specs.version.rpartition('.')[0]
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/jobs.{ver}.sqlite')

    def create_job(self, worker, user, payload=None):
        job_uid = gws.u.random_string(64)
        gws.log.debug(f'JOB {job_uid}: creating: {worker=}  {user.uid=}')

        self._db().insert(self.TABLE, dict(uid=job_uid))

        self._write(
            job_uid,
            dict(
                uid=job_uid,
                userUid=user.uid,
                userStr=self.root.app.authMgr.serialize_user(user),
                worker=sys.modules.get(worker.__module__).__file__ + '.' + worker.__name__,
                state=gws.JobState.open,
                payload=payload,
                created=gws.u.stime(),
                updated=gws.u.stime(),
            ),
        )

        return self._get_job_or_fail(job_uid)

    def get_job(self, job_uid: str, user=None, state=None):
        job, msg = self._get_job(job_uid, user, state)
        if not job:
            gws.log.error(msg)
        return job

    def _get_job_or_fail(self, job_uid: str, user=None, state=None):
        job, msg = self._get_job(job_uid, user, state)
        if not job:
            raise gws.Error(msg)
        return job

    def _get_job(self, job_uid: str, user=None, state=None):
        rs = self._db().select(f'SELECT * FROM {self.TABLE} WHERE uid=:uid', uid=job_uid)
        if not rs:
            return None, f'JOB {job_uid}: not found'

        rec = rs[0]

        job_user = None
        us = rec.get('userStr')
        if us:
            job_user = self.root.app.authMgr.unserialize_user(us)
        if not job_user:
            job_user = self.root.app.authMgr.guestUser

        if user and job_user.uid != user.uid:
            return None, f'JOB {job_uid}: wrong user {job_user.uid=} {user.uid=}'

        if state and rec.get('state') != state:
            return None, f'JOB {job_uid}: wrong state {rec.get("state")=} {state=}'

        job = gws.Job(
            uid=rec['uid'],
            user=job_user,
            worker=rec['worker'],
            state=rec['state'],
            error=rec['error'],
            numSteps=rec['numSteps'] or 0,
            step=rec['step'] or 0,
            stepName=rec['stepName'] or '',
            payload=gws.lib.jsonx.from_string(rec['payload'] or '{}'),
            timeCreated=dtx.from_timestamp(rec['created'] or 0),
            timeUpdated=dtx.from_timestamp(rec['updated'] or 0),
        )
        return job, ''

    def update_job(self, job, **kwargs):
        job = self.get_job(job.uid)
        if not job:
            return
        self._write(job.uid, kwargs)
        return self._get_job_or_fail(job.uid)

    def _write(self, job_uid, rec):
        rec['updated'] = gws.u.stime()
        p = rec.get('payload')
        if p is not None and not isinstance(p, str):
            rec['payload'] = gws.lib.jsonx.to_string(p)
        gws.log.debug(f'JOB {job_uid}: save {rec=}')
        self._db().update(self.TABLE, rec, job_uid)

    def schedule_job(self, job: gws.Job):
        job = self._get_job_or_fail(job.uid, state=gws.JobState.open)
        if gws.server.spool.is_active():
            gws.server.spool.add(job)
            return job
        return self.run_job(job)

    def run_job(self, job: gws.Job):
        job_uid = job.uid

        # atomically mark an 'open' job as 'running'

        tmp = gws.u.random_string(64)
        self._db().execute(
            f'UPDATE {self.TABLE} SET state=:tmp WHERE uid=:uid AND state=:state',
            uid=job_uid,
            tmp=tmp,
            state=gws.JobState.open,
        )
        job = self._get_job_or_fail(job_uid, state=tmp)
        self._db().execute(
            f'UPDATE {self.TABLE} SET state=:s WHERE uid=:uid',
            uid=job.uid,
            s=gws.JobState.running,
        )
        job = self._get_job_or_fail(job_uid, state=gws.JobState.running)

        # now it's ours, let's run it

        try:
            mod_path, _, fn_name = job.worker.rpartition('.')
            mod = gws.lib.importer.import_from_path(mod_path)
            worker_cls = getattr(mod, fn_name)
            worker_cls.run(self.root, job)
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
        if not job:
            raise gws.NotFoundError(f'JOB {p.jobUid}: not found')
        return self.job_status_response(job)

    def job_status_response(self, job, **kwargs):
        d = dict(
            jobUid=job.uid,
            state=job.state,
            stepName=job.stepName or '',
            progress=self._get_progress(job),
            output={},
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
