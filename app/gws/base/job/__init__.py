"""Background jobs.

Runs long tasks, such as printing or exporting, outside of the web request
that started them. The client starts a job, receives its uid and polls its
status until the job is complete, then fetches the result.

Submodules
----------

- ``manager`` - the job manager (``root.app.jobMgr``). It stores jobs in a
  SQLite database, schedules and runs them and handles status and cancel
  requests.
- ``worker`` - the base class for job workers, with access to the job record
  for progress updates and cancellation checks.

Jobs
----

A job record (``gws.Job``) holds the user, the worker class, the state,
progress counters, a payload for the worker and the result. Records live in
``jobs.<version>.sqlite`` in the misc directory, so that all server processes
see the same jobs.

A job goes through the states ``open`` (created), ``running`` and then
``complete``, ``error`` or ``cancel``. ``schedule_job`` passes the job to the
uWSGI spooler if it is available, otherwise the job runs at once in the
current process. ``run_job`` marks the job as running atomically, so a job runs
only once, imports the worker class and calls its ``run`` class method. An
exception in the worker, other than ``gws.JobTerminated``, puts the job into
the ``error`` state.

Workers
-------

A worker is a class with a ``run(root, job)`` class method, usually a subclass
of ``worker.Object``. While working, it reports progress with ``update_job``
and finally stores the result. ``get_job`` and ``update_job`` raise
``gws.JobTerminated`` when the job is no longer running, for example because
it was cancelled, which ends the worker.

Jobs belong to the user who created them; status, cancel and result requests
from other users are answered with ``gws.NotFoundError``.

Example::

    class MyWorker(gws.base.job.worker.Object):
        @classmethod
        def run(cls, root, job):
            w = cls(root, job.user, job)
            w.work(job.payload)

        def work(self, payload):
            self.update_job(numSteps=len(payload['items']))
            for n, item in enumerate(payload['items'], 1):
                ...
                self.update_job(step=n)
            self.update_job(state=gws.JobState.complete, result={'count': n})

    mgr = root.app.jobMgr
    job = mgr.create_job(MyWorker, user, payload={'items': [...]})
    job = mgr.schedule_job(job)
    return mgr.job_status_response(job)
"""

from . import manager, worker
