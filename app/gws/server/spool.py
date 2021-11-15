import gws
import gws.server.uwsgi_module
import gws.lib.job

# from uwsgi
OK = -2
RETRY = -1
IGNORE = 0


def add(job):
    uwsgi = gws.server.uwsgi_module.load()
    gws.log.info("SPOOLING", job.uid)
    d = {b'job_uid': gws.to_bytes(job.uid)}
    getattr(uwsgi, 'spool')(d)


def run(root: gws.IRoot, env: dict):
    job_uid = env.get(b'job_uid')
    if not job_uid:
        raise ValueError('no job_uid found')
    gws.lib.job.run(root, gws.to_str(job_uid))
