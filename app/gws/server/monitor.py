import re
import os
import psutil

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
    def configure(self):
        super().configure()
        self.watch_dirs = {}
        self.watch_files = {}
        self.path_stats = {}
        self.enabled = self.var('enabled', default=True)
        self.frequency = self.var('frequency', default=30)
        self.ignore = self.var('ignore', default=[])

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
        if not self.enabled:
            gws.log.info(f'MONITOR: disabled')
            return

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

        # only one worker is allowed to do that
        uwsgi.register_signal(42, 'worker1', self._worker)
        uwsgi.add_timer(42, self.frequency)
        gws.log.info(f'MONITOR: started, frequency={self.frequency}')

    def _worker(self, signo):
        with gws.tools.misc.lock(_lockfile) as ok:
            if not ok:
                try:
                    pid = int(gws.read_file(_lockfile))
                except:
                    pid = None
                if not pid or not psutil.pid_exists(pid):
                    gws.log.info(f'MONITOR: locked by dead pid={pid!r}, releasing')
                else:
                    gws.log.info(f'MONITOR: locked by pid={pid!r}')
                    return

            gws.write_file(_lockfile, str(os.getpid()))
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
        new_stats = {}
        changed = []

        for dirname, pattern in self.watch_dirs.items():
            if not self._ignored(dirname):
                for filename in gws.tools.os2.find_files(dirname, pattern):
                    if not self._ignored(filename):
                        new_stats[filename] = self._stats(filename)

        for filename, _ in self.watch_files.items():
            if filename not in new_stats and not self._ignored(filename):
                new_stats[filename] = self._stats(filename)

        for p in set(self.path_stats) | set(new_stats):
            if self.path_stats.get(p) != new_stats.get(p):
                changed.append(p)

        self.path_stats = new_stats
        return changed

    def _stats(self, path):
        try:
            s = os.stat(path)
            return s.st_size, s.st_mtime
        except OSError:
            return 0, 0

    def _ignored(self, filename):
        return self.ignore and any(re.search(p, filename) for p in self.ignore)
