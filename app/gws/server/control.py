import shlex
import time

import gws
import gws.config
import gws.lib.date
import gws.lib.osx
import gws.types as t

from . import ini

_START_SCRIPT = gws.VAR_DIR + '/server.sh'


def start(manifest_path=None, config_path=None):
    stop()
    root = configure(manifest_path, config_path, is_starting=True)
    gws.config.store(root)
    gws.config.activate(root)
    return start_configured()


def start_configured():
    for p in gws.lib.osx.find_files(gws.SERVER_DIR, '.*'):
        gws.lib.osx.unlink(p)

    commands = ini.create(gws.config.root(), gws.SERVER_DIR)

    with open(_START_SCRIPT, 'wt') as fp:
        fp.write('echo "----------------------------------------------------------"\n')
        fp.write('echo "SERVER START"\n')
        fp.write('echo "----------------------------------------------------------"\n')
        fp.write('\n'.join(commands))

    return _START_SCRIPT


def stop():
    _stop(['uwsgi', 'qgis_mapserv.fcgi', 'nginx'], signals=['INT', 'KILL'])
    _stop(['rsyslogd'], signals=['KILL'])


def configure_and_store(manifest_path=None, config_path=None):
    root = configure(manifest_path, config_path, is_starting=False)
    gws.config.store(root)


def reconfigure(manifest_path=None, config_path=None):
    if not _uwsgi_is_running():
        gws.log.info('server not running, starting...')
        start(manifest_path, config_path)
        return

    root = configure(manifest_path, config_path, is_starting=False)
    gws.config.store(root)
    reload()


def configure(manifest_path, config_path, is_starting=False):
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


def reload(modules=None):
    if not _uwsgi_is_running():
        return False
    for m in ('qgis', 'mapproxy', 'web', 'spool'):
        if not modules or m in modules:
            _reload_uwsgi(m)
    return True


##

def _reload_uwsgi(module):
    pattern = f'({module}).uwsgi.pid'
    for p in gws.lib.osx.find_files(gws.TMP_DIR, pattern):
        gws.log.info(f'reloading {p}...')
        gws.lib.osx.run(['/usr/local/bin/uwsgi', '--reload', p])


def _uwsgi_is_running():
    return bool(gws.lib.osx.pids_of('uwsgi'))


_FALLBACK_CONFIG = {
    'server': {
        'mapproxy': {'enabled': False},
        'monitor': {'enabled': False},
        'log': {'level': 'INFO'},
        'qgis': {'enabled': False},
        'spool': {'enabled': False},
        'web': {'enabled': True, 'workers': 1},
        'autoRun': '',
        'timeout': 60,
        'timeZone': 'UTC',
    }
}

_STOP_RETRY = 20
_STOP_PAUSE = 1


def _stop(names, signals):
    for sig in signals:
        for _ in range(_STOP_RETRY):
            if all(_stop_name(name, sig) for name in names):
                return
            time.sleep(_STOP_PAUSE)

    err = ''

    for name in names:
        pids = gws.lib.osx.pids_of(name)
        if pids:
            err += f' {name}={pids!r}'

    if err:
        raise ValueError(f'failed to stop processes: {err}')


def _stop_name(proc_name, sig):
    pids = gws.lib.osx.pids_of(proc_name)
    if not pids:
        return True
    for pid in pids:
        gws.log.debug(f'stopping {proc_name} pid={pid}')
        gws.lib.osx.kill_pid(pid, sig)
    return False
