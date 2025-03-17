"""Base job worker"""

from typing import Optional
import gws


class Object:
    def __init__(self, root: gws.Root, user: gws.User, job: Optional[gws.Job] = None):
        self.jobUid = job.uid if job else None
        self.root = root
        self.user = user

    def get_job(self) -> Optional[gws.Job]:
        if not self.jobUid:
            return

        job = self.root.app.jobMgr.get_job(
            self.jobUid,
            user=self.user,
            state=gws.JobState.running
        )
        if not job:
            self.jobUid = 0
            raise gws.JobTerminated(f'JOB {self.jobUid!r} TERMINATED')
        return job

    def update_job(self, **kwargs):
        job = self.get_job()
        if job:
            self.root.app.jobMgr.update_job(job, **kwargs)
