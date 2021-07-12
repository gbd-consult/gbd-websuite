import importlib

import gws


def add(job):
    uwsgi = importlib.import_module('uwsgi')
    gws.log.info("SPOOLING", job.uid)
    d = {b'job_uid': gws.as_bytes(job.uid)}
    getattr(uwsgi, 'spool')(d)
