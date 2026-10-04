"""Base job worker."""

from typing import Optional
import gws


class Object:
    """Base job worker.

    Subclasses implement a ``run(root, job)`` class method, which the job
    manager calls to run the job. A worker can also run without a job, then
    job updates are skipped.
    """

    jobUid: str
    """Uid of the job, empty if the worker runs without a job or the job was terminated."""
    user: gws.User
    """User the job runs for."""

    def __init__(self, root: gws.Root, user: gws.User, job: Optional[gws.Job] = None):
        """Create a worker.

        Args:
            root: Root object.
            user: User the job runs for.
            job: The job, or ``None`` to run without a job.
        """
        self.jobUid = job.uid if job else ''
        self.root = root
        self.user = user

    def get_job(self) -> Optional[gws.Job]:
        """Return the current state of the job.

        Returns:
            The job, or ``None`` if the worker runs without a job.

        Raises:
            ``gws.JobTerminated``: If the job no longer exists or is not running, e.g. because it was cancelled.
        """
        if not self.jobUid:
            return

        job = self.root.app.jobMgr.get_job(
            self.jobUid,
            user=self.user,
            state=gws.JobState.running
        )
        if not job:
            self.jobUid = ''
            raise gws.JobTerminated('JOB TERMINATED')
        return job

    def update_job(self, **kwargs):
        """Update the job record. Does nothing if the worker runs without a job.

        Args:
            **kwargs: Job fields to update, e.g. ``state``, ``step``, ``numSteps``, ``stepName`` or ``result``.

        Raises:
            ``gws.JobTerminated``: If the job no longer exists or is not running.
        """
        job = self.get_job()
        if job:
            self.root.app.jobMgr.update_job(job, **kwargs)
