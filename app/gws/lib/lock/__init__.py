"""Utilities for locking"""

import gws

import os
import fcntl
import time


class SoftFileLock:
    """Soft file-based locking.

    Attempt to lock a file and yield the success status.
    """

    def __init__(self, path: str):
        self.path = path
        self.fp = None

    def __enter__(self):
        self.fp = _lock(self.path)
        if not self.fp:
            return False
        os.write(self.fp, str(os.getpid()).encode('ascii'))
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp:
            _unlock(self.path, self.fp)
            self.fp = None


class HardFileLock:
    """Hard file-based locking.

    Keep attempting to lock a file until the timeout expires.
    """

    def __init__(self, path: str, retries: int = 10, pause: int = 2):
        self.path = path
        self.fp = None
        self.retries = retries
        self.pause = pause

    def __enter__(self):
        for _ in range(self.retries):
            self.fp = _lock(self.path)
            if self.fp:
                return True
            time.sleep(self.pause)
        raise gws.Error(f'failed to lock {self.path!r}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp:
            _unlock(self.path, self.fp)
            self.fp = None


def _lock(path):
    try:
        fp = os.open(path, os.O_CREAT | os.O_RDWR)
    except OSError as exc:
        gws.log.warning(f'lock: {path!r} open error: {exc}')
        return

    try:
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            pid = os.read(fp, 1024).decode('ascii')
        except OSError as exc:
            gws.log.warning(f'lock: {path!r} read error: {exc}')
        else:
            gws.log.info(f'lock: {path!r} locked by {pid}')
        return

    try:
        os.write(fp, str(os.getpid()).encode('ascii'))
    except OSError as exc:
        gws.log.warning(f'lock: {path!r} write error: {exc}')
        try:
            os.close(fp)
        except OSError:
            pass
    else:
        return fp


def _unlock(path, fp):
    try:
        fcntl.flock(fp, fcntl.LOCK_UN)
    except OSError as exc:
        gws.log.warning(f'lock: {path!r} unlock error: {exc}')
    try:
        os.close(fp)
    except OSError as exc:
        gws.log.warning(f'lock: {path!r} close error: {exc}')
    try:
        os.unlink(path)
    except OSError as exc:
        gws.log.warning(f'lock: {path!r} unlink error: {exc}')
