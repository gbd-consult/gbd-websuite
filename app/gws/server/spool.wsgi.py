import gws.config.loader
import gws
import gws.lib.job

# noinspection PyUnresolvedReferences
import uwsgi


def application(environ, start_response):
    pass


def _spooler(env):
    job_uid = env.get(b'job_uid')
    if not job_uid:
        raise ValueError('no job_uid found')
    job = gws.lib.job.get(gws.as_str(job_uid))
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


try:
    root = gws.config.loader.load()
    gws.log.set_level(root.application.var('server.logLevel'))
    root.application.monitor.start()
    uwsgi.spooler = spooler
except:
    gws.log.error('UNABLE TO LOAD CONFIGURATION')
    gws.log.exception()
    gws.exit(255)
