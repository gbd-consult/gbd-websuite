import gws
import gws.config
import gws.server.uwsgi_module

from . import runner


def application(environ, start_response):
    pass


def spooler(env):
    try:
        runner.run(gws.config.get_root(), env)
    except:
        gws.log.exception()

    # even if it's failed, return OK so the spooler can clean up
    # if we ever provide retry, this will on the app level, no automatic spooler retries
    return runner.OK


def init():
    root = None

    try:
        gws.log.info('starting SPOOL application')
        root = gws.config.load()
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.u.exit(1)

    gws.server.uwsgi_module.load().spooler = spooler

    try:
        gws.log.set_level(root.app.cfg('server.log.level'))
        root.app.monitor.start()
    except:
        gws.log.exception('SPOOL INIT ERROR')
        gws.u.exit(1)

