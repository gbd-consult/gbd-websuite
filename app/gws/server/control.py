"""Server control.

Following workflows are supported:

1) Server start. This is called only once upon the container start.

    - (empty TMP_DIR completely in bin/gws)
    - configure
    - store the config
    - write server configs
    - (the actual invocation of the server start script takes place in bin/gws)

2) Server reconfigure. Can be called anytime, e.g. by the monitor

    - configure
    - store the config
    - write server configs
    - empty the TRANSIENT_DIR
    - reload all uwsgis
    - reload nginx

3) Server reload. Can be called anytime, e.g. by the monitor

    - write server configs
    - empty the TRANSIENT_DIR
    - reload all uwsgis
    - reload nginx


4) Configure (debugging)

    - configure
    - store the config


5) Configtest (debugging)

    - configure



"""

import shlex
import time

import gws
import gws.config
import gws.lib.date
import gws.lib.osx
import gws.types as t

from . import ini

# see bin/gws
SERVER_START_SCRIPT = f'{gws.VAR_DIR}/server.sh'


def start(manifest_path=None, config_path=None):
    if server_is_running('web'):
        gws.log.error(f'server already running')
        gws.exit(1)
    root = configure_and_store(manifest_path, config_path, is_starting=True)
    ini.write_configs_and_start_script(root, gws.SERVER_DIR, SERVER_START_SCRIPT)


def reconfigure(manifest_path=None, config_path=None):
    if not server_is_running('web'):
        gws.log.error(f'server not running')
        gws.exit(1)
    root = configure_and_store(manifest_path, config_path, is_starting=False)
    ini.write_configs_and_start_script(root, gws.SERVER_DIR, SERVER_START_SCRIPT)
    reload_all()


def configure_and_store(manifest_path=None, config_path=None, is_starting=False):
    root = configure(manifest_path, config_path, is_starting)
    gws.config.store(root)
    return root


def configure(manifest_path=None, config_path=None, is_starting=False):
    def _before_init(cfg):
        autorun = gws.get(cfg, 'server.autoRun')
        if autorun:
            gws.log.info(f'AUTORUN: {autorun!r}')
            cmds = shlex.split(autorun)
            gws.lib.osx.run(cmds, echo=True)

        timezone = gws.get(cfg, 'server.timeZone')
        if timezone:
            gws.lib.date.set_system_time_zone(timezone)

    return gws.config.configure(
        manifest_path=manifest_path,
        config_path=config_path,
        before_init=_before_init if is_starting else None,
        fallback_config=_FALLBACK_CONFIG,
    )


##

def reload_all():
    gws.lib.osx.run(['rm', '-fr', gws.TRANSIENT_DIR])
    gws.ensure_system_dirs()
    for srv in ini.PID_PATHS:
        reload_server(srv)
    reload_nginx()
    return True


def reload_server(srv):
    if not server_is_running(srv):
        gws.log.debug(f'reload: {srv=} not running')
        return
    gws.log.info(f'reloading {srv}...')
    gws.lib.osx.run(['uwsgi', '--reload', ini.PID_PATHS[srv]])


def reload_nginx():
    gws.log.info(f'reloading nginx...')
    gws.lib.osx.run(['nginx', '-c', gws.SERVER_DIR + '/nginx.conf', '-s', 'reload'])


def server_is_running(srv):
    try:
        with open(ini.PID_PATHS[srv]) as fp:
            pid = int(fp.read())
    except (FileNotFoundError, ValueError):
        pid = 0
    gws.log.debug(f'found {pid=} for {srv=}')
    return pid and pid in gws.lib.osx.running_pids()


##


_FALLBACK_CONFIG = {
    'server': {
        'mapproxy': {'enabled': False},
        'monitor': {'enabled': False},
        'log': {'level': 'INFO'},
        'qgis': {'host': 'qgis', 'port': 80},
        'spool': {'enabled': False},
        'web': {'enabled': True, 'workers': 1},
        'autoRun': '',
        'timeout': 60,
        'timeZone': 'UTC',
    }
}
