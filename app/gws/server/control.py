import shlex
import time

import gws
import gws.spec.runtime
import gws.config
import gws.lib.date
import gws.lib.json2
import gws.lib.os2
import gws.types as t
from . import ini, types

_START_SCRIPT = gws.VAR_DIR + '/server.sh'


def start(manifest_path=None, config_path=None):
    stop()

    root = _configure(manifest_path, config_path, is_starting=True, with_spec_cache=True)
    gws.config.store(root)
    gws.config.activate(root)

    for p in gws.lib.os2.find_files(gws.SERVER_DIR, '.*'):
        gws.lib.os2.unlink(p)

    pid_dir = gws.ensure_dir('pids', gws.TMP_DIR)
    commands = ini.create(root, gws.SERVER_DIR, pid_dir)

    with open(_START_SCRIPT, 'wt') as fp:
        fp.write('echo "----------------------------------------------------------"\n')
        fp.write('echo "SERVER START"\n')
        fp.write('echo "----------------------------------------------------------"\n')
        fp.write('\n'.join(commands))


def stop():
    _stop(['uwsgi', 'qgis_mapserv.fcgi', 'nginx'], signals=['INT', 'KILL'])
    _stop(['rsyslogd'], signals=['KILL'])


def configure(manifest_path=None, config_path=None):
    root = _configure(manifest_path, config_path, is_starting=False, with_spec_cache=True)
    gws.config.store(root)


def reconfigure(manifest_path=None, config_path=None):
    if not _uwsgi_is_running():
        gws.log.info('server not running, starting...')
        start(manifest_path, config_path)
        return

    root = _configure(manifest_path, config_path, is_starting=False, with_spec_cache=True)
    gws.config.store(root)
    reload()


def reload(modules=None):
    if not _uwsgi_is_running():
        return False
    for m in ('qgis', 'mapproxy', 'web', 'spool'):
        if not modules or m in modules:
            _reload_uwsgi(m)
    return True


##

def _reload_uwsgi(module):
    pid_dir = gws.ensure_dir('pids', gws.TMP_DIR)
    pattern = f'({module}).uwsgi.pid'

    for p in gws.lib.os2.find_files(pid_dir, pattern):
        gws.log.info(f'reloading {p}...')
        gws.lib.os2.run(['uwsgi', '--reload', p])


def _uwsgi_is_running():
    return bool(gws.lib.os2.pids_of('uwsgi'))


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


def _configure(manifest_path, config_path, is_starting=False, with_spec_cache=True):
    def _before_init(cfg):
        autorun = gws.get(cfg, 'server.autoRun')
        if autorun:
            gws.log.info(f'AUTORUN: {autorun!r}')
            cmds = shlex.split(autorun)
            gws.lib.os2.run(cmds, echo=True)

        timezone = gws.get(cfg, 'server.timeZone')
        if timezone:
            gws.lib.date.set_system_time_zone(timezone)

    return gws.config.loader.configure_server(
        manifest_path=manifest_path,
        config_path=config_path,
        before_init=_before_init if is_starting else None,
        fallback_config= _FALLBACK_CONFIG,
        with_spec_cache=with_spec_cache
    )


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
        pids = gws.lib.os2.pids_of(name)
        if pids:
            err += f' {name}={pids!r}'

    if err:
        raise ValueError(f'failed to stop processes: {err}')


def _stop_name(proc_name, sig):
    pids = gws.lib.os2.pids_of(proc_name)
    if not pids:
        return True
    for pid in pids:
        gws.log.debug(f'stopping {proc_name} pid={pid}')
        gws.lib.os2.kill_pid(pid, sig)
    return False


##

class StartParams(gws.CliParams):
    config: t.Optional[str]  #: configuration file
    manifest: t.Optional[str]  #: manifest file


class ReloadParams(StartParams):
    modules: t.Optional[t.List[str]]  #: list of modules to reload ('qgis', 'mapproxy', 'web', 'spool')


@gws.ext.Object('cli.server')
class Cli:

    @gws.ext.command('cli.server.start')
    def start(self, p: StartParams):
        """Configure and start the server"""
        start(p.manifest, p.config)

    @gws.ext.command('cli.server.restart')
    def restart(self, p: StartParams):
        """Stop and start the server"""
        self.start(p)

    @gws.ext.command('cli.server.stop')
    def stop(self, p: gws.NoParams):
        """Stop the server"""
        stop()

    @gws.ext.command('cli.server.reload')
    def reload(self, p: ReloadParams):
        """Reload specific (or all) server modules"""
        if not reload(p.modules):
            gws.log.info('server not running, starting')
            self.start(t.cast(StartParams, p))

    @gws.ext.command('cli.server.reconfigure')
    def reconfigure(self, p: StartParams):
        """Reconfigure and restart the server"""
        reconfigure(p.manifest, p.config)

    @gws.ext.command('cli.server.configure')
    def configure(self, p: StartParams):
        """Configure the server, but do not restart"""
        configure(p.manifest, p.config)

    @gws.ext.command('cli.server.configtest')
    def configtest(self, p: StartParams):
        """Test the configuration"""
        _configure(p.manifest or '', p.config or '', is_starting=False, with_spec_cache=False)
