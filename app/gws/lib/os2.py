"""Utilities for os/shell scripting"""

import hashlib
import os
import re
import signal
import subprocess

import psutil

import gws
import gws.types as t


class Error(gws.Error):
    pass


class TimeoutError(Error):
    pass


def run_nowait(cmd, **kwargs):
    """Run a process and return immediately"""

    args = {
        'stdin': None,
        'stdout': None,
        'stderr': None,
        'shell': False,
    }
    args.update(kwargs)

    return subprocess.Popen(cmd, **args)


def run(cmd, input=None, echo=False, strict=True, timeout=None, **kwargs):
    """Run a process, return a tuple (rc, output)"""

    args = {
        'stdin': subprocess.PIPE if input else None,
        'stdout': None if echo else subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'shell': False,
    }
    args.update(kwargs)

    try:
        p = subprocess.Popen(cmd, **args)
        out, _ = p.communicate(input, timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        raise TimeoutError()
    except Exception as e:
        raise Error from e

    if rc and strict:
        raise Error('command failed', cmd, rc, out)

    return rc, out


def unlink(path):
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def rename(src, dst):
    os.replace(src, dst)


def file_mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0


def file_size(path):
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def file_checksum(path):
    try:
        with open(path, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
    except OSError:
        return '0'


def kill_pid(pid, sig_name='TERM'):
    sig = getattr(signal, sig_name, None) or getattr(signal, 'SIG' + sig_name)
    try:
        psutil.Process(pid).send_signal(sig)
        return True
    except psutil.NoSuchProcess:
        return True
    except psutil.Error as e:
        gws.log.warn(f'send_signal failed, pid={pid!r}, {e}')
        return False


def pids_of(proc_name):
    pids = []

    for p in psutil.process_iter():
        if p.name() == proc_name:
            pids.append(p.pid)

    return pids


def find_files(dirname, pattern=None, ext=None, deep=True):
    if not pattern and ext:
        if isinstance(ext, (list, tuple)):
            ext = '|'.join(ext)
        pattern = '\\.(' + ext + ')$'

    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue

        if de.is_dir() and deep:
            yield from find_files(de.path, pattern)
            continue

        if de.is_file() and (pattern is None or re.search(pattern, de.path)):
            yield de.path


def parse_path(path):
    """Parse a path into a dict(path,dirname,filename,name,extension)"""

    d = {'path': path}

    d['dirname'], d['filename'] = os.path.split(path)
    if d['filename'].startswith('.'):
        d['name'], d['extension'] = d['filename'], ''
    else:
        d['name'], _, d['extension'] = d['filename'].partition('.')

    return d


def abs_path(path, base):
    """Absolutize a relative path with respect to a base directory or file path"""

    if os.path.isabs(path):
        return path

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.abspath(os.path.join(base, path))


def rel_path(path, base):
    """Relativize an absolute path with respect to a base directory or file path"""

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.relpath(path, base)


def chown(path, user=None, group=None):
    try:
        os.chown(path, user or gws.UID, group or gws.GID)
    except OSError:
        pass
