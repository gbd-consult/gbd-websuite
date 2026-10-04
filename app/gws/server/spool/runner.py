"""Queue and run background jobs with the uWSGI spooler."""

import gws
import gws.server.uwsgi_module

# from uwsgi
OK = -2
RETRY = -1
IGNORE = 0


def is_active():
    """Check whether the uWSGI spooler is available.

    Returns:
        True if the code runs in a uWSGI process.
    """
    try:
        gws.server.uwsgi_module.load()
        return True
    except ModuleNotFoundError:
        return False


def add(job: gws.Job):
    """Queue a job for the spooler.

    Args:
        job: The job to queue. Only its uid is passed to the spooler.
    """
    uwsgi = gws.server.uwsgi_module.load()
    gws.log.info(f'SPOOL: {job.uid=} added')
    d = {b'job_uid': gws.u.to_bytes(job.uid)}
    getattr(uwsgi, 'spool')(d)


def run(root: gws.Root, env: dict):
    """Run a queued job.

    Logs an error if the job uid is missing or the job is not found.

    Args:
        root: Root object.
        env: Spooler arguments, as passed by uWSGI, with the job uid in ``b'job_uid'``.
    """
    job_uid = env.get(b'job_uid')
    if not job_uid:
        gws.log.error(f'no "job_uid"')
        return
    job = root.app.jobMgr.get_job(gws.u.to_str(job_uid))
    if not job:
        gws.log.error(f'{job_uid=} not found')
        return
    root.app.jobMgr.run_job(job)
