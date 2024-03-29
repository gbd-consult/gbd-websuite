"""Utilities for os/shell scripting"""

import hashlib
import os
import re
import signal
import subprocess
import time

import psutil

import gws
import gws.types as t


class Error(gws.Error):
    pass


class TimeoutError(Error):
    pass


def getenv(key: str, default: str = None) -> t.Optional[str]:
    return os.getenv(key, default)


def utime():
    return time.time()


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

    scmd = cmd
    if isinstance(cmd, list):
        scmd = ' '.join(str(s) for s in cmd)

    gws.log.debug(f'RUN: {scmd}')

    try:
        p = subprocess.Popen(cmd, **args)
        out, _ = p.communicate(input, timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        raise TimeoutError(f'command timed out', scmd)
    except Exception as exc:
        raise Error(f'command failed', scmd) from exc

    if rc and strict:
        gws.log.debug(f'OUT: {out}')
        gws.log.debug(f'RC:  {rc}')
        raise Error(f'non-zero exit', cmd)

    return rc, out


def unlink(path):
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def rename(src, dst):
    os.replace(src, dst)


def chown(path, user=None, group=None):
    try:
        os.chown(path, user or gws.UID, group or gws.GID)
    except OSError:
        pass


def file_mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return -1


def file_age(path):
    try:
        return int(time.time() - os.stat(path).st_mtime)
    except OSError:
        return -1


def file_size(path):
    try:
        return os.stat(path).st_size
    except OSError:
        return -1


def file_checksum(path):
    try:
        with open(path, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
    except OSError:
        return ''


def kill_pid(pid, sig_name='TERM'):
    sig = getattr(signal, sig_name, None) or getattr(signal, 'SIG' + sig_name)
    try:
        psutil.Process(pid).send_signal(sig)
        return True
    except psutil.NoSuchProcess:
        return True
    except psutil.Error as e:
        gws.log.warning(f'send_signal failed, pid={pid!r}, {e}')
        return False


def running_pids() -> dict[int, str]:
    d = {}
    for p in psutil.process_iter():
        d[p.pid] = p.name()
    return d


def process_rss_size(unit='m') -> float:
    n = psutil.Process().memory_info().rss
    if unit == 'k':
        return n / 1e3
    if unit == 'm':
        return n / 1e6
    if unit == 'g':
        return n / 1e9
    return n


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


def find_directories(dirname, pattern=None, deep=True):
    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue

        if not de.is_dir():
            continue

        if pattern is None or re.search(pattern, de.path):
            yield de.path

        if deep:
            yield from find_directories(de.path, pattern)


_Path = str | bytes


def parse_path(path: _Path) -> dict[str, str]:
    """Parse a path into a dict(path,dirname,filename,name,extension)"""

    str_path = path if isinstance(path, str) else path.decode('utf8')
    sp = os.path.split(str_path)

    d = {
        'dirname': sp[0],
        'filename': sp[1],
        'name': '',
        'extension': '',
    }

    if d['filename'].startswith('.'):
        d['name'] = d['filename']
    else:
        par = d['filename'].partition('.')
        d['name'] = par[0]
        d['extension'] = par[2]

    return d


def file_name(path: _Path) -> str:
    str_path = path if isinstance(path, str) else path.decode('utf8')
    sp = os.path.split(str_path)
    return sp[1]


def abs_path(path: _Path, base: str) -> str:
    """Absolutize a relative path with respect to a base directory or file path"""

    str_path = path if isinstance(path, str) else path.decode('utf8')

    if os.path.isabs(str_path):
        return str_path

    if not base:
        raise ValueError('cannot compute abspath without a base')

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.abspath(os.path.join(base, str_path))


def abs_web_path(path: str, basedir: str) -> t.Optional[str]:
    """Return an absolute path in a base dir and ensure the path is correct."""

    _dir_re = r'^[A-Za-z0-9_]+$'
    _fil_re = r'^[A-Za-z0-9_]+(\.[a-z0-9]+)*$'

    gws.log.debug(f'abs_web_path: trying {path!r} in {basedir!r}')

    dirs = []
    for s in path.split('/'):
        s = s.strip()
        if s:
            dirs.append(s)

    fname = dirs.pop()

    if not all(re.match(_dir_re, p) for p in dirs):
        gws.log.error(f'abs_web_path: invalid dirname in path={path!r}')
        return

    if not re.match(_fil_re, fname):
        gws.log.error(f'abs_web_path: invalid filename in path={path!r}')
        return

    p = basedir
    if dirs:
        p += '/' + '/'.join(dirs)
    p += '/' + fname

    if not os.path.isfile(p):
        gws.log.error(f'abs_web_path: not a file path={path!r}')
        return

    return p


def rel_path(path: _Path, base: str) -> str:
    """Relativize an absolute path with respect to a base directory or file path"""

    if os.path.isfile(base):
        base = os.path.dirname(base)

    str_path = path if isinstance(path, str) else path.decode('utf8')

    return os.path.relpath(str_path, base)
