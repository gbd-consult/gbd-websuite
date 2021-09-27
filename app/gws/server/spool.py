import gws
import gws.server.uwsgi_module

# from uwsgi
OK = -2
RETRY = -1
IGNORE = 0


def add(job):
    uwsgi = gws.server.uwsgi_module.load()
    gws.log.info("SPOOLING", job.uid)
    d = {b'job_uid': gws.as_bytes(job.uid)}
    getattr(uwsgi, 'spool')(d)
