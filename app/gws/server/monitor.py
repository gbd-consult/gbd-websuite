import os

import gws
import gws.core.tree
import gws.config
import gws.tools.misc
import gws.tools.os2

import gws.types as t

from . import control

try:
    # noinspection PyUnresolvedReferences
    import uwsgi
except:
    pass

_lockfile = '/tmp/monitor.lock'


def _m() -> t.IMonitor:
    return gws.config.root().application.monitor


#:export IMonitor
class Object(gws.Object, t.IMonitor):
    def __init__(self):
        super().__init__()
        self.watch_dirs = {}
        self.watch_files = {}
        self.cpaths = {}

    def add_directory(self, path, pattern):
        if os.path.isfile(path):
            path = os.path.dirname(path)
        self.watch_dirs[path] = pattern

    def add_path(self, path):
        if os.path.isfile(path):
            self.watch_files[path] = 1
        else:
            raise ValueError(f'{path!r} is not a file')

    def start(self):
        # @TODO: use file monitor
        # actually, we should be using uwsgi.add_file_monitor here, however I keep having problems
        # getting inotify working on docker-mounted volumes (is that possible at all)?

        self._prepare()

        for s in self.watch_dirs:
            gws.log.info(f'MONITOR: watching directory {s!r}')
        for s in self.watch_files:
            gws.log.info(f'MONITOR: watching file {s!r}')

        try:
            os.unlink(_lockfile)
        except OSError:
            pass

        self._poll()

        freq = self.root.var('server.spool.monitorFrequency')
        # only one worker is allowed to do that
        uwsgi.register_signal(42, 'worker1', self._worker)
        uwsgi.add_timer(42, freq)
        gws.log.info(f'MONITOR: started, frequency={freq}')

    def _worker(self, signo):
        with gws.tools.misc.lock(_lockfile) as ok:
            if not ok:
                gws.log.info('MONITOR: locked...')
                return

            changed = self._poll()

            if not changed:
                return

            for path in changed:
                gws.log.info(f'MONITOR: changed {path!r}')

            # @TODO: smarter reload

            reconf = any(gws.APP_DIR not in path for path in changed)

            gws.log.info(f'MONITOR: begin reload (reconfigure={reconf})')

            if not self._reload(reconf):
                return

        # finally, reload ourselves
        gws.log.info(f'MONITOR: bye bye')
        control.reload_uwsgi('spool')

    def _reload(self, reconf):
        if reconf:
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
            return True
        except Exception:
            gws.log.error('MONITOR: reload error')
            gws.log.exception()
            return False

    def _prepare(self):
        dirs = self.watch_dirs
        ds = []
        for d in sorted(dirs):
            if d in ds or any(d.startswith(r + '/') for r in ds):
                continue
            ds.append(d)
        self.watch_dirs = {d: dirs[d] for d in ds}

    def _poll(self):
        paths = {}
        changed = []

        for dirname, pattern in self.watch_dirs.items():
            for filename in gws.tools.os2.find_files(dirname, pattern):
                paths[filename] = self._stats(filename)

        for filename, _ in self.watch_files.items():
            if filename not in paths:
                paths[filename] = self._stats(filename)

        for p in set(self.cpaths) | set(paths):
            if self.cpaths.get(p) != paths.get(p):
                changed.append(p)

        self.cpaths = paths
        return changed

    def _stats(self, path):
        try:
            s = os.stat(path)
            return s.st_size, s.st_mtime
        except OSError:
            return 0, 0
