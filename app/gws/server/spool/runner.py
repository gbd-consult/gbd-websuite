import gws
import gws.server.uwsgi_module

# from uwsgi
OK = -2
RETRY = -1
IGNORE = 0


def is_active():
    try:
        gws.server.uwsgi_module.load()
        return True
    except ModuleNotFoundError:
        return False


def add(job: gws.Job):
    uwsgi = gws.server.uwsgi_module.load()
    gws.log.info(f'SPOOL: {job.uid=} added')
    d = {b'job_uid': gws.u.to_bytes(job.uid)}
    getattr(uwsgi, 'spool')(d)


def run(root: gws.Root, env: dict):
    job_uid = env.get(b'job_uid')
    if not job_uid:
        gws.log.error(f'no "job_uid"')
        return
    job = root.app.jobMgr.get_job(gws.u.to_str(job_uid))
    if not job:
        gws.log.error(f'{job_uid=} not found')
        return
    root.app.jobMgr.run_job(job)
