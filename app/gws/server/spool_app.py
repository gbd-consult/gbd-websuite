import gws
import gws.config
import gws.lib.job
import gws.server.spool
import gws.server.uwsgi_module

def application(environ, start_response):
    pass


def spooler(env):
    try:
        gws.server.spool.run(gws.config.root(), env)
    except:
        gws.log.exception()

    # even if it's failed, return OK so the spooler can clean up
    # if we ever provide retry, this will on the app level, no automatic spooler retries
    return gws.server.spool.OK


def init():
    root = None

    try:
        gws.log.info('starting SPOOL application')
        root = gws.config.load()
    except:
        gws.log.exception('UNABLE TO LOAD CONFIGURATION')
        gws.exit(255)

    try:
        gws.log.set_level(root.application.var('server.log.level'))
        root.application.monitor.start()
    except:
        gws.log.exception('SPOOL INIT ERROR')
        gws.exit(255)

    gws.server.uwsgi_module.load().spooler = spooler

# NB: spooler must be inited right away

init()
