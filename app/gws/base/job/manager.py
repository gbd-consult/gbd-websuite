"""Job manager."""

from typing import Optional

import gws
import gws.lib.importer
import gws.lib.jsonx
import gws.lib.sqlitex
import gws.lib.mime
import gws.server.spool


class Request(gws.Request):
    jobUid: str


class Object(gws.JobManager):
    TABLE = 'jobs'
    DDL = f'''
        CREATE TABLE IF NOT EXISTS {TABLE} (
            uid           TEXT NOT NULL PRIMARY KEY,
            user_uid      TEXT NOT NULL,
            str_user      TEXT NOT NULL,
            worker        TEXT NOT NULL,
            state         TEXT DEFAULT '',
            error         TEXT DEFAULT '',
            numSteps      INTEGER DEFAULT 0,
            step          INTEGER DEFAULT 0,
            step_name     TEXT DEFAULT '',
            stepName      TEXT DEFAULT '',
            requestPath   TEXT DEFAULT '',
            outputPath    TEXT DEFAULT '',
            outputMime    TEXT DEFAULT '',
            extra         TEXT DEFAULT '',
            created       INTEGER NOT NULL,
            updated       INTEGER NOT NULL
        )
    '''

    dbPath: str

    def configure(self):
        self.dbPath = self.cfg('path', default=f'{gws.c.MISC_DIR}/jobs82.sqlite')

    def create_job(self, user, worker):
        job_uid = gws.u.random_string(64)
        gws.log.debug(f'JOB {job_uid}: creating: {worker=}  {user.uid=}')

        rec = dict(
            uid=job_uid,
            user_uid=user.uid,
            str_user=self.root.app.authMgr.serialize_user(user),
            worker=worker,
            state=gws.JobState.open,
            created=gws.u.stime(),
            updated=gws.u.stime()
        )

        self._db().insert(self.TABLE, rec)
        return self.get_job(job_uid)

    def get_job(self, job_uid: str, user=None):
        rs = self._db().select(f'SELECT * FROM {self.TABLE} WHERE uid=:uid', uid=job_uid)
        if not rs:
            gws.log.error(f'JOB {job_uid}: not found')
            return

        rec = rs[0]

        job_user = None
        if rec.get('str_user'):
            job_user = self.root.app.authMgr.unserialize_user(rec.get('str_user'))
        if not job_user:
            job_user = self.root.app.authMgr.guestUser

        if user and job_user.uid != user.uid:
            gws.log.error(f'JOB {job_uid}: wrong user {job_user.uid=} {user.uid=}')
            return

        return gws.Job(
            uid=rec['uid'],
            user=job_user,
            worker=rec['worker'],
            state=rec['state'],
            error=rec['error'],
            numSteps=rec['numSteps'] or 0,
            step=rec['step'] or 0,
            stepName=rec['stepName'] or '',
            requestPath=rec['requestPath'] or '',
            outputPath=rec['outputPath'] or '',
            outputMime=rec['outputMime'] or '',
            extra=gws.lib.jsonx.from_string(rec['extra'] or '{}'),
        )

    def save_job(self, job):
        if not self.get_job(job.uid):
            return

        rec = dict(
            state=job.state,
            error=job.error,
            numSteps=job.numSteps,
            step=job.step,
            stepName=job.stepName,
            requestPath=job.requestPath,
            outputPath=job.outputPath,
            outputMime=job.outputMime,
            extra=gws.lib.jsonx.to_string(job.extra or {}),
            updated=gws.u.stime(),
        )

        self._db().update(self.TABLE, rec, job.uid)
        return self.get_job(job.uid)

    def schedule_job(self, job: gws.Job):
        if gws.server.spool.is_active():
            gws.server.spool.add(job)
            return job
        return self.run_job(job)

    def run_job(self, job: gws.Job):
        job_uid = job.uid
        tmp = gws.u.random_string(64)

        self._db().execute(
            f'UPDATE {self.TABLE} SET state=:s WHERE uid=:uid AND state=:state',
            uid=job_uid, s=tmp, state=gws.JobState.open
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

    ##

    def status_response(self, job):
        return gws.JobResponse(
            jobUid=job.uid,
            state=job.state,
            progress=self._get_progress(job),
            stepName=job.stepName or '',
            outputUrl=self._get_output_url(job),
        )

    def _get_progress(self, job):
        if job.state == gws.JobState.complete:
            return 100
        if job.state != gws.JobState.running:
            return 0
        if not job.numSteps:
            return 0
        return int(min(100.0, job.step * 100.0 / job.numSteps))

    def _get_output_url(self, job):
        if job.state != gws.JobState.complete:
            return ''
        ext = gws.lib.mime.extension_for(job.outputMime) or 'bin'
        return gws.u.action_url_path('jobOutput', jobUid=job.uid) + '/gws.' + ext

    ##

    _sqlitex: gws.lib.sqlitex.Object

    def _db(self):
        if getattr(self, '_sqlitex', None) is None:
            self._sqlitex = gws.lib.sqlitex.Object(self.dbPath, self.DDL)
        return self._sqlitex
