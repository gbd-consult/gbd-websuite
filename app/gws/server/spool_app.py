# noinspection PyUnresolvedReferences
import uwsgi

import gws
import gws.config
import gws.lib.job

_inited: bool = False


def application(environ, start_response):
    global _inited

    if not _inited:
        _init()
        _inited = True


def _init():
    try:
        gws.log.info('starting SPOOL application')
        root = gws.config.load()
        gws.log.set_level(root.application.var('server.logLevel'))
        root.application.monitor.start()
        uwsgi.spooler = _spooler
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(255)


##


def _spooler(env):
    try:
        _spooler2(env)
    except:
        gws.log.exception()

    # even if it's failed, return OK so the spooler can clean up
    # if we ever provide retry, this will on the app level, no automatic spooler retries

    return gws.SPOOL_OK


def _spooler2(env):
    job_uid = env.get(b'job_uid')
    if not job_uid:
        raise ValueError('no job_uid found')
    job = gws.lib.job.get(gws.as_str(job_uid))
    if not job:
        raise ValueError('invalid job_uid', job_uid)
    gws.log.debug('running job', job.uid)
    job.run()
