import shlex
import time

import gws
import gws.config.loader
import gws.lib.date
import gws.lib.json2
import gws.lib.mpx.config
import gws.lib.os2
import gws.types as t
from . import ini

_START_SCRIPT = gws.VAR_DIR + '/server.sh'


def start(manifest_path=None, config_path=None):
    stop()

    root = _configure(manifest_path, config_path, is_starting=True)
    gws.config.loader.store(root)
    gws.config.loader.activate(root)

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
    _stop('uwsgi')
    _stop('nginx')
    _stop('qgis_mapserv.fcgi')
    _stop('rsyslogd')


def configure(manifest_path=None, config_path=None, store=False):
    root = _configure(manifest_path, config_path, is_starting=False)
    if store:
        gws.config.loader.store(root)


def reconfigure(manifest_path=None, config_path=None):
    pid = gws.lib.os2.pids_of('uwsgi')
    if not pid:
        gws.log.info('server not running, starting...')
        start(manifest_path, config_path)
        return

    root = _configure(manifest_path, config_path, is_starting=False)
    gws.config.loader.store(root)

    for m in ('qgis', 'mapproxy', 'web', 'spool'):
        reload_uwsgi(m)


def reload(modules=None):
    _reload(False, None, modules)


def reload_uwsgi(module):
    pid_dir = gws.ensure_dir('pids', gws.TMP_DIR)
    pattern = f'({module}).uwsgi.pid'

    for p in gws.lib.os2.find_files(pid_dir, pattern):
        gws.log.info(f'reloading {p}...')
        gws.lib.os2.run(['uwsgi', '--reload', p])


def _configure(manifest_path, config_path, is_starting):
    cfg = gws.config.loader.parse(manifest_path, config_path)

    if is_starting:
        autorun = gws.get(cfg, 'server.autoRun')
        if autorun:
            gws.log.info(f'AUTORUN: {autorun!r}')
            cmds = shlex.split(autorun)
            gws.lib.os2.run(cmds, echo=True)

        timezone = gws.get(cfg, 'server.timeZone')
        if timezone:
            gws.lib.date.set_system_time_zone(timezone)

    root = gws.config.loader.initialize(cfg)

    if root.application.var('server.mapproxy.enabled'):
        gws.lib.mpx.config.create_and_save(root, ini.MAPPROXY_YAML_PATH)

    gws.log.info('CONFIGURATION OK')
    return root


def _stop(proc_name):
    if _kill_name(proc_name, 'INT'):
        return

    for _ in range(10):
        if _kill_name(proc_name, 'KILL'):
            return
        time.sleep(5)

    pids = gws.lib.os2.pids_of(proc_name)
    if pids:
        raise ValueError(f'failed to stop {proc_name} pids={pids!r}')


def _kill_name(proc_name, sig_name):
    pids = gws.lib.os2.pids_of(proc_name)
    if not pids:
        return True
    for pid in pids:
        gws.log.debug(f'stopping {proc_name} pid={pid}')
        gws.lib.os2.kill_pid(pid, sig_name)
    return False


##

class StartParams(gws.CliParams):
    config: t.Optional[str]


class DumpParams(gws.CliParams):
    config: t.Optional[str]
    out: t.Optional[str]


class ReloadParams(gws.CliParams):
    modules: t.Optional[t.List[str]]


@gws.ext.Object('cli.server')
class Cli(gws.Object):

    @gws.ext.command('cli.server.start')
    def start(self, p: StartParams):
        return start(p.manifest, p.config)

    @gws.ext.command('cli.server.stop')
    def stop(self, p: gws.NoParams):
        return stop()

    @gws.ext.command('cli.server.configure')
    def configure(self, p: StartParams):
        return configure(p.manifest, p.config, store=True)

    @gws.ext.command('cli.server.configtest')
    def configtest(self, p: StartParams):
        return configure(p.manifest, p.config, store=False)

    @gws.ext.command('cli.server.configdump')
    def configdump(self, p: DumpParams):
        if p.config:
            root = gws.config.loader.initialize(gws.config.loader.parse(p.manifest, p.config))
        else:
            root = gws.config.loader.load()

        json = gws.lib.json2.to_tagged_string(root, pretty=True, ascii=False)
        if p.out:
            gws.write_file(p.out, json)
        else:
            print(json)
