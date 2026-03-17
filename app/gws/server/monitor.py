import os

import gws
import gws.config
import gws.lib.lock
import gws.lib.osx
import gws.lib.watcher
import gws.server.uwsgi_module

from . import control

_LOCK_FILE = '/tmp/monitor.lock'
_RELOAD_FILE = '/tmp/monitor.reload'
_RECONFIGURE_FILE = '/tmp/monitor.reconfigure'
_TICK_FREQUENCY = 3

DEFAULT_FREQUENCY = 30


class _Task(gws.Data):
    obj: gws.Node
    frequency: int
    lastTime: int


class Object(gws.ServerMonitor):
    watchPaths: set[str]
    enabled: bool
    frequency: int
    watcher: gws.lib.watcher.Watcher
    dirs: list
    files: list
    tasks: list[_Task]

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
        os.open(_RECONFIGURE_FILE if with_reconfigure else _RELOAD_FILE, os.O_CREAT | os.O_WRONLY)

    def start(self):
        self._check_unlink(_LOCK_FILE)
        self._check_unlink(_RELOAD_FILE)
        self._check_unlink(_RECONFIGURE_FILE)

        for t in self.tasks:
            t.lastTime = gws.u.stime()

        if not self.cfg('disableWatch'):

            def notify(evt, path):
                os.open(_RECONFIGURE_FILE, os.O_CREAT | os.O_WRONLY)

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
        do_reconfigure = self._check_unlink(_RECONFIGURE_FILE)
        do_reload = self._check_unlink(_RELOAD_FILE)

        tasks = [t for t in self.tasks if gws.u.stime() - t.lastTime >= t.frequency]

        if not do_reconfigure and not do_reload and not tasks:
            # gws.log.debug(f'MONITOR: tick skip')
            return

        gws.log.debug(f'MONITOR: tick {do_reconfigure=} {do_reload=} tasks={[t.obj for t in tasks]}')

        try:
            os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
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
        gws.log.info(f'MONITOR: reloading...')

        if not self._reload2(with_reconfigure):
            return

        # ok, reload ourselves
        gws.log.info(f'MONITOR: bye bye')
        control.reload_app('spool')

    def _reload2(self, with_reconfigure):
        if with_reconfigure:
            try:
                control.configure_and_store()
            except Exception as exc:
                gws.log.exception(f'MONITOR: configuration error {exc!r}')
                return False

        try:
            control.reload_app('mapproxy')
            control.reload_app('web')
            return True
        except Exception as exc:
            gws.log.exception(f'MONITOR: reload error {exc!r}')
            return False

    def _run_periodic_tasks(self, tasks):
        for t in tasks:
            try:
                t.obj.periodic_task()
                t.lastTime = gws.u.stime()
            except:
                gws.log.exception(f'MONITOR: periodic task failed {t.obj}')

    def _check_unlink(self, path):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return False
