import gws
import importlib


def add(job):
    uwsgi = importlib.import_module('uwsgi')
    gws.log.info("SPOOLING", job.uid)
    d = {b'job_uid': gws.as_bytes(job.uid)}
    uwsgi.spool(d)
