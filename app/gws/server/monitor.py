import os
import re

import gws
import gws.config
import gws.lib.lock
import gws.lib.osx
import gws.server.uwsgi_module

from . import control

_LOCK_FILE = '/tmp/monitor.lock'


class Object(gws.Node, gws.IMonitor):
    watchDirs: dict
    watchFiles: dict
    pathStats: dict

    enabled: bool
    frequency: int
    ignore: list[str]

    def configure(self):
        self.enabled = self.cfg('enabled', default=True)
        self.frequency = self.cfg('frequency', default=30)
        self.ignore = self.cfg('ignore', default=[])

        self.watchDirs = {}
        self.watchFiles = {}
        self.pathStats = {}

    def add_directory(self, path, pattern):
        self.watchDirs[path] = pattern

    def add_file(self, path):
        self.watchFiles[path] = 1

    def start(self):
        if not self.enabled:
            gws.log.info(f'MONITOR: disabled')
            return

        # @TODO: use file monitor
        # actually, we should be using uwsgi.add_file_monitor here, however I keep having problems
        # getting inotify working on docker-mounted volumes (is that possible at all)?

        self._cleanup()

        for s in self.watchDirs:
            gws.log.info(f'MONITOR: watching directory {s!r}')
        for s in self.watchFiles:
            gws.log.info(f'MONITOR: watching file {s!r}')

        try:
            os.unlink(_LOCK_FILE)
        except OSError:
            pass

        self._poll()

        uwsgi = gws.server.uwsgi_module.load()

        # only one worker is allowed to do that
        uwsgi.register_signal(42, 'worker2', self._worker)
        uwsgi.add_timer(42, self.frequency)
        gws.log.info(f'MONITOR: started, frequency={self.frequency}')

    def _worker(self, signo):
        with gws.lib.lock.SoftFileLock(_LOCK_FILE) as ok:
            if not ok:
                return

            changed_paths = self._poll()
            if not changed_paths:
                return

            for path in changed_paths:
                gws.log.info(f'MONITOR: changed {path!r}')

            # @TODO: smarter reload

            needs_reconfigure = any(not path.endswith('.py') for path in changed_paths)
            gws.log.info(f'MONITOR: begin reload {needs_reconfigure=}')

            if not self._reload(needs_reconfigure):
                return

        # finally, reload ourselves
        gws.log.info(f'MONITOR: bye bye')
        control.reload_server('spool')

    def _reload(self, needs_reconfigure):
        if needs_reconfigure:
            try:
                control.configure_and_store()
            except:
                gws.log.exception('MONITOR: configuration error')
                return False

        try:
            control.reload_server('mapproxy')
            control.reload_server('web')
            # reloading nginx in a spooler doesn't work properly,
            # but actually it is not needed
            # control.reload_nginx()
            return True
        except:
            gws.log.exception('MONITOR: reload error')
            return False

    def _cleanup(self):
        """Remove superfluous directory and file entries."""

        ls = []
        for d in sorted(self.watchDirs):
            if d in ls or any(d.startswith(e + '/') for e in ls):
                # if we watch /some/dir already, there's no need to watch /some/dir/subdir
                continue
            ls.append(d)
        self.watchDirs = {d: self.watchDirs[d] for d in sorted(ls)}

        ls = []
        for f in sorted(self.watchFiles):
            if any(f.startswith(e + '/') for e in self.watchDirs):
                # if we watch /some/dir already, there's no need to watch /some/dir/some.file
                continue
            ls.append(f)
        self.watchFiles = {f: 1 for f in sorted(ls)}

    def _poll(self):
        new_stats = {}
        changed_paths = []

        for dirpath, pattern in self.watchDirs.items():
            if self._ignored(dirpath):
                continue
            if not gws.is_dir(dirpath):
                continue
            for path in gws.lib.osx.find_files(dirpath, pattern):
                if not self._ignored(path):
                    new_stats[path] = self._stats(path)

        for path, _ in self.watchFiles.items():
            if path not in new_stats and not self._ignored(path):
                new_stats[path] = self._stats(path)

        for path in set(self.pathStats) | set(new_stats):
            if self.pathStats.get(path) != new_stats.get(path):
                changed_paths.append(path)

        self.pathStats = new_stats
        return changed_paths

    def _stats(self, path):
        try:
            s = os.stat(path)
            return s.st_size, s.st_mtime
        except OSError:
            return 0, 0

    def _ignored(self, filename):
        return self.ignore and any(re.search(p, filename) for p in self.ignore)
