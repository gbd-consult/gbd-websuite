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


def rel_path(path: _Path, base: str) -> str:
    """Relativize an absolute path with respect to a base directory or file path"""

    if os.path.isfile(base):
        base = os.path.dirname(base)

    str_path = path if isinstance(path, str) else path.decode('utf8')

    return os.path.relpath(str_path, base)
