"""File system watcher.

Monitor file system events such as file creation, deletion, modification, and movement.
The module allows directories or specific files to be monitored, supports pattern matching for files,
and provides a mechanism to trigger custom notification callbacks when events occur.
"""

from typing import Callable, TypeAlias
import os
import re

import watchdog.events
import watchdog.observers

import gws

_WATCH_EVENTS = {
    watchdog.events.EVENT_TYPE_MOVED,
    watchdog.events.EVENT_TYPE_DELETED,
    watchdog.events.EVENT_TYPE_CREATED,
    watchdog.events.EVENT_TYPE_MODIFIED,
}

_EVENTS = []


class _DirEntry:
    def __init__(self, dirname, pattern, recursive):
        self.dirname = dirname
        self.pattern = pattern
        self.recursive = recursive


_NotifyFn: TypeAlias = Callable[[str, str], None]


def new(notify: _NotifyFn):
    """Create a new watcher.

    Args:
        notify: A callback function that will be called when an event occurs.
            It should accept two arguments: the event type and the path of the file.
            Note that the callback will be called from a different thread than the one that created the watcher.
    """
    return Watcher(notify)


class Watcher:
    observer: watchdog.observers.Observer

    def __init__(self, notify: _NotifyFn):
        self.notify = notify
        self.dirEntries = {}
        self.filePaths = set()
        self.excludePatterns = []

    def add_directory(self, dirname: str | os.PathLike, file_pattern: str = '', recursive: bool = False):
        d = str(dirname)
        self.dirEntries[d] = _DirEntry(d, file_pattern or '.', recursive)

    def add_file(self, filename: str | os.PathLike):
        self.filePaths.add(str(filename))

    def exclude(self, path_pattern: str):
        self.excludePatterns.append(path_pattern)

    def start(self):
        self.observer = watchdog.observers.Observer()

        h = _Handler(self)

        for de in self.dirEntries.values():
            gws.log.debug(f'watcher: watching {de.dirname!r}')
            self.observer.schedule(h, de.dirname, recursive=de.recursive)

        for f in self.filePaths:
            gws.log.debug(f'watcher: watching {f!r}')
            self.observer.schedule(h, os.path.dirname(f), recursive=False)

        self.observer.start()
        gws.log.debug(f'watcher: started with {self.observer.__class__.__name__}')

    def stop(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            gws.log.debug(f'watcher: stopped')

    def register(self, ev: watchdog.events.FileSystemEvent):
        if self.path_matches(ev.src_path):
            gws.log.debug(f'watcher: {ev.event_type} {ev.src_path}')
            self.notify(ev.event_type, ev.src_path)

    def path_matches(self, path):
        if any(re.search(ex, path) for ex in self.excludePatterns):
            return False
        if path in self.filePaths:
            return True
        d, f = os.path.split(path)
        for de in self.dirEntries.values():
            if (d == de.dirname or (de.recursive and d.startswith(de.dirname + '/'))) and re.search(de.pattern, f):
                return True
        return False


class _Handler(watchdog.events.FileSystemEventHandler):
    def __init__(self, obj: Watcher):
        super().__init__()
        self.obj = obj

    def on_any_event(self, ev):
        if ev.event_type in _WATCH_EVENTS:
            self.obj.register(ev)
