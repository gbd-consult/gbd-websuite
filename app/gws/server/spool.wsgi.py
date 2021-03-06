import gws.config.loader
import gws
import gws.tools.job

# noinspection PyUnresolvedReferences
import uwsgi


def application(environ, start_response):
    pass


def _spooler(env):
    job_uid = env.get(b'job_uid')
    if not job_uid:
        raise ValueError('no job_uid found')
    job = gws.tools.job.get(gws.as_str(job_uid))
    if not job:
        raise ValueError('invalid job_uid', job_uid)
    gws.log.debug('running job', job.uid)
    job.run()


def spooler(env):
    try:
        _spooler(env)
    except:
        gws.log.exception()

    # even if it's failed, return OK so the spooler can clean up
    # if we ever provide retry, this will on the app level, no automatic spooler retries

    return gws.SPOOL_OK


root = gws.config.loader.load()
root.application.monitor.start()
uwsgi.spooler = spooler
