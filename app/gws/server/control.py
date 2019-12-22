import time

import gws
import gws.config.loader
import gws.gis.mpx.config
import gws.tools.date
import gws.tools.json2
import gws.tools.misc as misc
import gws.tools.shell as sh
import gws.types
from . import ini

commands_path = gws.VAR_DIR + '/server.sh'
pid_dir = misc.ensure_dir('pids', gws.TMP_DIR)


def start():
    stop()

    root = configure()
    gws.tools.date.set_system_time_zone(root.var('timeZone'))

    for p in misc.find_files(gws.SERVER_DIR, '.*'):
        sh.unlink(p)

    commands = ini.create(root, gws.SERVER_DIR, pid_dir)

    s = root.var('server.autoRun')
    if s:
        commands.insert(0, s)

    with open(commands_path, 'wt') as fp:
        fp.write('echo "----------------------------------------------------------"\n')
        fp.write('echo "SERVER START"\n')
        fp.write('echo "----------------------------------------------------------"\n')
        fp.write('\n'.join(commands))


def stop():
    _stop('uwsgi')
    _stop('nginx')
    _stop('qgis_mapserv.fcgi')


def _stop(proc_name):
    _kill_name(proc_name, 'INT')

    for _ in range(10):
        time.sleep(5)
        if _kill_name(proc_name, 'KILL'):
            return

    pids = sh.pids_of(proc_name)
    if pids:
        raise ValueError(f'failed to stop {proc_name} pids={pids!r}')


def reload():
    _reload(True)


def reset(module=None):
    _reload(False, module)


def reload_uwsgi(module):
    pattern = f'({module}).uwsgi.pid'
    for p in misc.find_files(pid_dir, pattern):
        gws.log.info(f'reloading {p}...')
        sh.run(['uwsgi', '--reload', p])


def configure():
    root = gws.config.loader.parse_and_activate()
    if root.var('server.mapproxy.enabled'):
        gws.gis.mpx.config.create_and_save(root, ini.MAPPROXY_YAML_PATH)
    gws.config.loader.store()
    gws.log.info('CONFIGURATION OK')
    return root


def _reload(reconfigure, module=None):
    pid = sh.pids_of('uwsgi')
    if not pid:
        gws.log.info('server not running, starting...')
        return start()

    if reconfigure:
        configure()

    for m in ('qgis', 'mapproxy', 'web'):
        if not module or m == module:
            reload_uwsgi(m)


def _kill_name(proc_name, sig_name):
    pids = sh.pids_of(proc_name)
    if not pids:
        return True
    for pid in pids:
        gws.log.info(f'stopping {proc_name} pid={pid}')
        sh.kill_pid(pid, sig_name)
    return False
