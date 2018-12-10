import os
import time

import gws
import gws.core.tree
import gws.config
import gws.tools.misc as misc

from . import control

try:
    # noinspection PyUnresolvedReferences
    import uwsgi
except:
    pass

# monitor data is saved in the config, so the monitor must be a gws.Object

class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.watch_dirs = {}
        self.watch_files = {}
        self.cpaths = {}


def _m() -> Object:
    return gws.config.root().monitor


def add_directory(path, pattern):
    if os.path.isfile(path):
        path = os.path.dirname(path)
    _m().watch_dirs[path] = pattern


def add_path(path):
    if os.path.isfile(path):
        _m().watch_files[path] = 1
    else:
        raise ValueError(f'{path!r} is not a file')


def start():
    # @TODO: use file monitor
    # actually, we should be using uwsgi.add_file_monitor here, however I keep having problems
    # getting inotify working on docker-mounted volumes (is that possible at all)?

    _prepare()

    for s in _m().watch_dirs:
        gws.log.info(f'MONITOR: watching directory {s!r}...')
    for s in _m().watch_files:
        gws.log.info(f'MONITOR: watching file {s!r}...')

    _poll()

    freq = gws.config.var('server.spool.monitorFrequency')
    # only one worker is allowed to do that
    uwsgi.register_signal(42, 'worker1', _worker)
    uwsgi.add_timer(42, freq)
    gws.log.info(f'MONITOR: started, frequency={freq}')


_lockfile = '/tmp/monitor.lock'


def _worker(signo):
    with misc.lock(_lockfile) as ok:
        if not ok:
            gws.log.info('MONITOR: locked...')
            return

        changed = _poll()

        if not changed:
            return

        for path in changed:
            gws.log.info(f'MONITOR: changed {path!r}')

        # @TODO: smarter reload

        gws.log.info('MONITOR: begin reload')

        if not _reload():
            gws.log.info('MONITOR: reload failed')
            return

        # @TODO: check when reload complete
        time.sleep(30)
        _poll()
        gws.log.info('MONITOR: end reload')


def _reload():
    try:
        control.configure()
    except Exception:
        gws.log.error('MONITOR: configuration error')
        gws.log.exception()
        return False

    try:
        control.reload_uwsgi('qgis')
        control.reload_uwsgi('mapproxy')
        control.reload_uwsgi('web')
    except Exception:
        gws.log.error('MONITOR: reload error')
        gws.log.exception()
        return False

    return True


def _prepare():
    dirs = _m().watch_dirs
    ds = []
    for d in sorted(dirs):
        if d in ds or any(d.startswith(r + '/') for r in ds):
            continue
        ds.append(d)
    _m().watch_dirs = {d: dirs[d] for d in ds}


def _poll():
    m = _m()
    paths = {}
    changed = []

    for dirname, pattern in m.watch_dirs.items():
        for filename in misc.find_files(dirname, pattern):
            paths[filename] = _stats(filename)

    for filename, _ in m.watch_files.items():
        if filename not in paths:
            paths[filename] = _stats(filename)

    for p in set(m.cpaths) | set(paths):
        if m.cpaths.get(p) != paths.get(p):
            changed.append(p)

    m.cpaths = paths
    return changed


def _stats(path):
    try:
        s = os.stat(path)
        return s.st_size, s.st_mtime
    except OSError:
        return 0, 0
