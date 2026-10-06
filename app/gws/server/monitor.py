"""Server monitor, which watches files and runs periodic tasks."""

import os

import gws
import gws.lib.watcher
import gws.server.uwsgi_module

from . import control

_LOCK_FILE = '/tmp/monitor.lock'
_RELOAD_FILE = '/tmp/monitor.reload'
_RECONFIGURE_FILE = '/tmp/monitor.reconfigure'
_TICK_FREQUENCY = 3

DEFAULT_FREQUENCY = 30


class _Task(gws.Data):
    """A registered periodic task."""

    obj: gws.Node
    """Object with a ``periodic_task`` method."""
    frequency: int
    """Interval between runs in seconds."""
    lastTime: int
    """Time of the last run, as a Unix timestamp."""


class Object(gws.ServerMonitor):
    """Server monitor."""

    enabled: bool
    """Not used."""
    frequency: int
    """Default interval of periodic tasks in seconds."""
    watcher: gws.lib.watcher.Watcher
    """File watcher, created on start unless ``disableWatch`` is set."""
    dirs: list
    """Watched directories, as tuples of directory, file pattern and recursive flag."""
    files: list
    """Watched files."""
    tasks: list[_Task]
    """Registered periodic tasks."""

    def configure(self):
        self.frequency = self.cfg('frequency', default=DEFAULT_FREQUENCY)
        self.dirs = []
        self.files = []
        self.tasks = []

    def watch_directory(self, dirname, pattern, recursive=False):
        self.dirs.append((dirname, pattern, recursive))

    def watch_file(self, path):
        self.files.append(path)

    def register_periodic_task(self, obj, frequency=0):
        if not hasattr(obj, 'periodic_task'):
            raise gws.Error(f'MONITOR: {obj!r} has no periodic_task')
        self.tasks.append(
            _Task(
                obj=obj,
                frequency=frequency or self.frequency,
                lastTime=0,
            )
        )

    def schedule_reload(self, with_reconfigure=False):
        gws.log.info(f'MONITOR: reload scheduled {with_reconfigure=}')
        self._touch(_RECONFIGURE_FILE if with_reconfigure else _RELOAD_FILE)

    def start(self):
        self._check_unlink(_LOCK_FILE)
        self._check_unlink(_RELOAD_FILE)
        self._check_unlink(_RECONFIGURE_FILE)

        for t in self.tasks:
            t.lastTime = gws.u.stime()

        if not self.cfg('disableWatch'):

            def notify(evt, path):
                self._touch(_RECONFIGURE_FILE)

            self.watcher = gws.lib.watcher.new(notify)

            for d in self.dirs:
                self.watcher.add_directory(*d)
            for f in self.files:
                self.watcher.add_file(f)

            self.watcher.start()

        uwsgi = gws.server.uwsgi_module.load()
        uwsgi.register_signal(42, 'worker2', self._tick)
        uwsgi.add_timer(42, _TICK_FREQUENCY)

        gws.log.info(f'MONITOR: started')

    def _tick(self, signo):
        """Handle the timer signal: reconfigure, reload or run the periodic tasks that are due.

        Reconfigure and reload requests are signalled through marker files in ``/tmp``,
        so that they reach the monitor from any process. A lock file prevents overlapping runs.
        """
        do_reconfigure = self._check_unlink(_RECONFIGURE_FILE)
        do_reload = self._check_unlink(_RELOAD_FILE)

        tasks = [t for t in self.tasks if gws.u.stime() - t.lastTime >= t.frequency]

        if not do_reconfigure and not do_reload and not tasks:
            # gws.log.debug(f'MONITOR: tick skip')
            return

        gws.log.debug(f'MONITOR: tick {do_reconfigure=} {do_reload=} tasks={len(tasks)}')

        try:
            self._touch(_LOCK_FILE, excl=True)
        except FileExistsError:
            gws.log.debug(f'MONITOR: locked')
            return

        try:
            if do_reconfigure:
                self._reload(True)
            elif do_reload:
                self._reload(False)
            elif tasks:
                self._run_periodic_tasks(tasks)
        finally:
            self._check_unlink(_LOCK_FILE)

    def _reload(self, with_reconfigure):
        """Reload the web backend, then the spool backend, which restarts the monitor.

        If the configuration fails, nothing is reloaded.
        """
        gws.log.info(f'MONITOR: reloading...')

        if not self._reload2(with_reconfigure):
            return

        # ok, reload ourselves
        gws.log.info(f'MONITOR: bye bye')
        control.reload_app('spool')

    def _reload2(self, with_reconfigure):
        """Optionally reconfigure, then reload the web backend, return False on failure."""
        if with_reconfigure:
            try:
                control.configure_and_store()
            except Exception as exc:
                gws.log.exception(f'MONITOR: configuration error: {exc!r}')
                return False

        try:
            control.reload_app('web')
            return True
        except Exception as exc:
            gws.log.exception(f'MONITOR: reload error: {exc!r}')
            return False

    def _run_periodic_tasks(self, tasks):
        """Run the given tasks, logging and skipping failed ones."""
        for t in tasks:
            t.lastTime = gws.u.stime()
            try:
                t.obj.periodic_task()
            except Exception as exc:
                gws.log.exception(f'MONITOR: periodic task failed {t.obj}: {exc!r}')

    def _touch(self, path, excl=False):
        """Create an empty file, fail if it exists and ``excl`` is set."""
        flags = os.O_CREAT | os.O_WRONLY
        if excl:
            flags |= os.O_EXCL
        os.close(os.open(path, flags))

    def _check_unlink(self, path):
        """Delete a file, return True if it existed."""
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return False
