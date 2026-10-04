"""File system watcher.

Monitors file system events (creation, deletion, modification and movement) using ``watchdog``,
and calls a notification callback when an event occurs on a watched path.

A ``Watcher`` can watch whole directories, optionally recursively and restricted to file names
matching a regular expression, and individual files. Paths matching an exclude pattern are ignored.
The callback is called from the observer thread, not from the thread that started the watcher.

Example::

    import gws.lib.watcher

    def on_change(event_type, path):
        print(event_type, path)

    w = gws.lib.watcher.new(on_change)
    w.add_directory('/data/config', file_pattern=r'[.]json$', recursive=True)
    w.add_file('/data/config.cx')
    w.exclude(r'/[.]git/')
    w.start()
    ...
    w.stop()
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
    """A watched directory."""

    def __init__(self, dirname, pattern, recursive):
        self.dirname = dirname
        self.pattern = pattern
        self.recursive = recursive


_NotifyFn: TypeAlias = Callable[[str, str], None]


def new(notify: _NotifyFn):
    """Create a new watcher.

    Args:
        notify: A callback function that will be called when an event occurs.
            It accepts two arguments: the event type and the path of the file.
            The callback is called from a different thread than the one that created the watcher.

    Returns:
        The watcher, not started yet.
    """
    return Watcher(notify)


class Watcher:
    """File system watcher."""

    observer: watchdog.observers.Observer
    """The ``watchdog`` observer, created by ``start``."""

    def __init__(self, notify: _NotifyFn):
        """Create a watcher.

        Args:
            notify: Callback, called with the event type and the path.
        """
        self.notify = notify
        self.dirEntries = {}
        self.filePaths = set()
        self.excludePatterns = []

    def add_directory(self, dirname: str | os.PathLike, file_pattern: str = '', recursive: bool = False):
        """Add a directory to watch.

        Args:
            dirname: Directory path.
            file_pattern: Regular expression to search for in file names. If empty, all files match.
            recursive: Also watch subdirectories.
        """
        d = str(dirname)
        self.dirEntries[d] = _DirEntry(d, file_pattern or '.', recursive)

    def add_file(self, filename: str | os.PathLike):
        """Add a file to watch.

        Args:
            filename: File path.
        """
        self.filePaths.add(str(filename))

    def exclude(self, path_pattern: str):
        """Exclude paths from watching.

        Args:
            path_pattern: Regular expression to search for in the full path.
        """
        self.excludePatterns.append(path_pattern)

    def start(self):
        """Start watching the added directories and files in a background thread."""
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
        """Stop watching and wait for the observer thread to finish."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            gws.log.debug(f'watcher: stopped')

    def register(self, ev: watchdog.events.FileSystemEvent):
        """Handle a file system event, calling the callback if the path is watched.

        Args:
            ev: The event.
        """
        if self.path_matches(ev.src_path):
            gws.log.debug(f'watcher: {ev.event_type} {ev.src_path}')
            self.notify(ev.event_type, ev.src_path)

    def path_matches(self, path):
        """Check if a path is watched.

        A path is watched if it is not excluded, and is either an added file
        or a file in an added directory whose name matches the directory's file pattern.

        Args:
            path: File path.

        Returns:
            ``True`` if the path is watched.
        """
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
    """Event handler that passes watched event types to the watcher."""

    def __init__(self, obj: Watcher):
        super().__init__()
        self.obj = obj

    def on_any_event(self, ev):
        if ev.event_type in _WATCH_EVENTS:
            self.obj.register(ev)
