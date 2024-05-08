"""Utilities for os/shell scripting"""

from typing import Optional

import hashlib
import os
import re
import signal
import subprocess
import time

import psutil

import gws


class Error(gws.Error):
    pass


class TimeoutError(Error):
    pass


_Path = str | bytes


def getenv(key: str, default: str = None) -> Optional[str]:
    """Returns the value for a given environment-variable.

    Args:
        key: An environment-variable.
        default: The default return.

    Returns:
        ``default`` if no key has been found, if there is such key then the value for the environment-variable is returned.
        """
    return os.getenv(key, default)


def utime() -> float:
    """Returns the time in seconds since the Epoch."""
    return time.time()


def run_nowait(cmd: str, **kwargs) -> subprocess.Popen:
    """Run a process and return immediately.

    Args:
        cmd: A process to run.
        kwargs:

    Returns:
        The output of the command.
    """

    args = {
        'stdin': None,
        'stdout': None,
        'stderr': None,
        'shell': False,
    }
    args.update(kwargs)

    return subprocess.Popen(cmd, **args)


def run(cmd: str | list, input: str = None, echo: bool = False,
        strict: bool = True, timeout: float = None, **kwargs) -> tuple:
    """Run a process, return a tuple (rc, output).

    Args:
        cmd: Command to run.
        input:
        echo:
        strict:
        timeout:
        kwargs:

    Returns:
        ``(rc,output)``
    """

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


def unlink(path: _Path):
    """Deletes a given path.

    Args:
        path: Filepath.
    """
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def rename(src: _Path, dst: _Path):
    """Moves and renames the source path according to the given destination.

    Args:
        src: Path to source.
        dst: Destination.
    """
    os.replace(src, dst)


def chown(path: _Path, user: int = None, group: int = None):
    """Changes the UID or GID for a given path.

    Args:
        path: Filepath.
        user: UID.
        group: GID.
    """
    try:
        os.chown(path, user or gws.c.UID, group or gws.c.GID)
    except OSError:
        pass


def file_mtime(path: _Path) -> float:
    """Returns the time from epoch when the path was recently changed.

    Args:
        path: File-/directory-path.

    Returns:
        Time since epoch in seconds until most recent change in file.
    """
    try:
        return os.stat(path).st_mtime
    except OSError:
        return -1


def file_age(path: _Path) -> int:
    """Returns the amount of seconds since the path has been changed.

    Args:
        path: Filepath.

    Returns:
        Amount of seconds since most recent change in file, if the path is invalid ``-1`` is returned.
    """
    try:
        return int(time.time() - os.stat(path).st_mtime)
    except OSError:
        return -1


def file_size(path: _Path) -> int:
    """Returns the file size.

    Args:
        path: Filepath.

    Returns:
        Amount of characters in the file or ``-1`` if the path is invalid.
    """
    try:
        return os.stat(path).st_size
    except OSError:
        return -1


def file_checksum(path: _Path) -> str:
    """Reuturs the checksum of the file.

    Args:
        path: Filepath.

    Returns:
        Empty string if the path is invalid, otherwise the file's checksum.
    """
    try:
        with open(path, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
    except OSError:
        return ''


def kill_pid(pid: int, sig_name='TERM') -> bool:
    """Kills a process.

    Args:
        pid: Process ID.
        sig_name:

    Returns:
        ``True`` if the process with the given PID is killed or does not exist.``False `` if the process could not be killed.
        """
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
    """Returns the current pids and the corresponding process' name."""
    d = {}
    for p in psutil.process_iter():
        d[p.pid] = p.name()
    return d


def process_rss_size(unit: str = 'm') -> float:
    """Returns the Resident Set Size.

    Agrs:
        unit: ``m`` | ``k`` | ``g``

    Returns:
        The Resident Set Size with the given unit.
    """
    n = psutil.Process().memory_info().rss
    if unit == 'k':
        return n / 1e3
    if unit == 'm':
        return n / 1e6
    if unit == 'g':
        return n / 1e9
    return n


def find_files(dirname: _Path, pattern=None, ext=None, deep: bool = True):
    """Finds files in a given directory.

    Args:
        dirname: Path to directory.
        pattern: Pattern to match.
        ext: extension to match.
        deep: If true then searches through all subdirectories for files,
                otherwise it returns the files only in the given directory.

    Returns:
        A generator object.
    """
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


def find_directories(dirname: _Path, pattern=None, deep: bool = True):
    """Finds all directories in a given directory.

    Args:
        dirname: Path to directory.
        pattern: Pattern to match.
        deep: If true then searches through all subdirectories for directories,
                otherwise it returns the directories only in the given directory.

    Returns:
        A generator object.
    """
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


def parse_path(path: _Path) -> dict[str, str]:
    """Parse a path into a dict(path,dirname,filename,name,extension).

    Args:
        path: Path.

    Returns:
        A dict(path,dirname,filename,name,extension).
    """

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
    """Returns the filename.

    Args:
        path: Filepath.

    Returns:
        The filename.
    """
    str_path = path if isinstance(path, str) else path.decode('utf8')
    sp = os.path.split(str_path)
    return sp[1]


def is_abs_path(path: _Path) -> bool:
    return os.path.isabs(path)


def abs_path(path: _Path, base: str) -> str:
    """Absolutize a relative path with respect to a base directory or file path.

    Args:
        path: A path.
        base: A path to the base.

    Raises:
        ``ValueError``: If base is empty

    Returns:
        The absolutized path.
    """

    str_path = path if isinstance(path, str) else path.decode('utf8')

    if os.path.isabs(str_path):
        return str_path

    if not base:
        raise ValueError('cannot compute abspath without a base')

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.abspath(os.path.join(base, str_path))


def abs_web_path(path: str, basedir: str) -> Optional[str]:
    """Return an absolute path in a base dir and ensure the path is correct.

    Args:
        path: Path to absolutize.
        basedir: Path to base directory.

    Returns:
        Absolutized path with respect to base directory.
    """

    _dir_re = r'^[A-Za-z0-9_-]+$'
    _fil_re = r'^[A-Za-z0-9_-]+(\.[a-z0-9]+)*$'

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
    """Relativize an absolute path with respect to a base directory or file path.

    Args:
        path: Path to relativize.
        base: Path to base directory.

    Returns:
        Relativized path with respect to base directory.

    """

    if os.path.isfile(base):
        base = os.path.dirname(base)

    str_path = path if isinstance(path, str) else path.decode('utf8')

    return os.path.relpath(str_path, base)
