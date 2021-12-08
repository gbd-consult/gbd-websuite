import time
import shlex

import gws
import gws.config.loader
import gws.gis.mpx.config
import gws.tools.date
import gws.tools.json2
import gws.tools.os2

from . import ini

_START_SCRIPT = gws.VAR_DIR + '/server.sh'


def configure(config_path=None, is_starting=False):
    cfg = gws.config.loader.parse(config_path)

    if is_starting:
        autorun = gws.get(cfg, 'server.autoRun')
        if autorun:
            gws.log.info(f'AUTORUN: {autorun!r}')
            cmds = shlex.split(autorun)
            gws.tools.os2.run(cmds, echo=True)

        timezone = gws.get(cfg, 'server.timeZone')
        if timezone:
            gws.tools.date.set_system_time_zone(timezone)

    root = gws.config.loader.activate(cfg)

    if root.var('server.mapproxy.enabled'):
        gws.gis.mpx.config.create_and_save(root, ini.MAPPROXY_YAML_PATH)

    gws.config.loader.store(root)

    gws.log.info('CONFIGURATION OK')
    return root


def start(config_path=None):
    stop()

    root = configure(config_path, is_starting=True)

    for p in gws.tools.os2.find_files(gws.SERVER_DIR, '.*'):
        gws.tools.os2.unlink(p)

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


def _stop(proc_name):
    if _kill_name(proc_name, 'INT'):
        return

    for _ in range(10):
        if _kill_name(proc_name, 'KILL'):
            return
        time.sleep(5)

    pids = gws.tools.os2.pids_of(proc_name)
    if pids:
        raise ValueError(f'failed to stop {proc_name} pids={pids!r}')


def reconfigure(config_path=None):
    _reload(True, config_path)


def reload(modules=None):
    _reload(False, None, modules)


def reload_uwsgi(module):
    pid_dir = gws.ensure_dir('pids', gws.TMP_DIR)
    pattern = f'({module}).uwsgi.pid'

    for p in gws.tools.os2.find_files(pid_dir, pattern):
        gws.log.info(f'reloading {p}...')
        gws.tools.os2.run(['uwsgi', '--reload', p])


def _reload(reconf, config_path, modules=None):
    pid = gws.tools.os2.pids_of('uwsgi')
    if not pid:
        gws.log.info('server not running, starting...')
        start(config_path)
        return

    if reconf:
        configure(config_path)

    for m in ('qgis', 'mapproxy', 'web', 'spool'):
        if not modules or m in modules:
            reload_uwsgi(m)


def _kill_name(proc_name, sig_name):
    pids = gws.tools.os2.pids_of(proc_name)
    if not pids:
        return True
    for pid in pids:
        gws.log.debug(f'stopping {proc_name} pid={pid}')
        gws.tools.os2.kill_pid(pid, sig_name)
    return False
