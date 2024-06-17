"""Utilities for os/shell scripting"""

from typing import Optional

import hashlib
import os
import re
import signal
import subprocess
import shutil
import shlex
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


def run_nowait(cmd: str | list, **kwargs) -> subprocess.Popen:
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


def run(cmd: str | list, input: str = None, echo: bool = False, strict: bool = True, timeout: float = None, **kwargs) -> str:
    """Run an external command.

    Args:
        cmd: Command to run.
        input: Input data.
        echo: Echo the output instead of capturing it.
        strict: Raise an error on a non-zero exit code.
        timeout: Timeout.
        kwargs: Arguments to pass to ``subprocess.Popen``.

    Returns:
        The command output.
    """

    args = {
        'stdin': subprocess.PIPE if input else None,
        'stdout': None if echo else subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'shell': False,
    }
    args.update(kwargs)

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    gws.log.debug(f'RUN: {cmd=}')

    try:
        p = subprocess.Popen(cmd, **args)
        out, _ = p.communicate(input, timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f'run: command timed out', repr(cmd)) from exc
    except Exception as exc:
        raise Error(f'run: failure', repr(cmd)) from exc

    if rc:
        gws.log.debug(f'RUN_FAILED: {cmd=} {rc=} {out=}')

    if rc and strict:
        raise Error(f'run: non-zero exit', repr(cmd))

    return _to_str(out or '')


def unlink(path: _Path) -> bool:
    """Deletes a given path.

    Args:
        path: Filepath.
    """
    try:
        if os.path.isfile(path):
            os.unlink(path)
        return True
    except OSError as exc:
        gws.log.warning(f'OSError: unlink: {exc}')
        return False


def rename(src: _Path, dst: _Path) -> bool:
    """Moves and renames the source path according to the given destination.

    Args:
        src: Path to source.
        dst: Destination.
    """

    try:
        os.replace(src, dst)
        return True
    except OSError as exc:
        gws.log.warning(f'OSError: rename: {exc}')
        return False


def chown(path: _Path, user: int = None, group: int = None) -> bool:
    """Changes the UID or GID for a given path.

    Args:
        path: Filepath.
        user: UID.
        group: GID.
    """
    try:
        os.chown(path, user or gws.c.UID, group or gws.c.GID)
        return True
    except OSError as exc:
        gws.log.warning(f'OSError: chown: {exc}')
        return False


def mkdir(path: _Path, mode: int = 0o755, user: int = None, group: int = None) -> bool:
    """Check a (possibly nested) directory.

    Args:
        path: Path to a directory.
        mode: Directory creation mode.
        user: Directory user (defaults to gws.c.UID)
        group: Directory group (defaults to gws.c.GID)
    """

    try:
        os.makedirs(path, mode, exist_ok=True)
        return chown(path, user, group)
    except OSError as exc:
        gws.log.warning(f'OSError: mkdir: {exc}')
        return False


def rmdir(path: _Path) -> bool:
    """Remove a directory or a directory tree.

    Args:
        path: Path to a directory. Can be non-empty
    """

    try:
        shutil.rmtree(path)
        return True
    except OSError as exc:
        gws.log.warning(f'OSError: rmdir: {exc}')
        return False


def file_mtime(path: _Path) -> float:
    """Returns the time from epoch when the path was recently changed.

    Args:
        path: File-/directory-path.

    Returns:
        Time since epoch in seconds until most recent change in file.
    """
    try:
        return os.stat(path).st_mtime
    except OSError as exc:
        gws.log.warning(f'OSError: file_mtime: {exc}')
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
    except OSError as exc:
        gws.log.warning(f'OSError: file_age: {exc}')
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
    except OSError as exc:
        gws.log.warning(f'OSError: file_size: {exc}')
        return -1


def file_checksum(path: _Path) -> str:
    """Returns the checksum of the file.

    Args:
        path: Filepath.

    Returns:
        Empty string if the path is invalid, otherwise the file's checksum.
    """
    try:
        with open(path, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
    except OSError as exc:
        gws.log.warning(f'OSError: file_checksum: {exc}')
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

    Args:
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

    str_path = _to_str(path)
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

    sp = os.path.split(_to_str(path))
    return sp[1]


def is_abs_path(path: _Path) -> bool:
    return os.path.isabs(path)


def abs_path(path: _Path, base: _Path) -> str:
    """Absolutize a relative path with respect to a base directory or file path.

    Args:
        path: A path.
        base: A path to the base.

    Raises:
        ``ValueError``: If base is empty

    Returns:
        The absolute path.
    """

    str_path = _to_str(path)

    if os.path.isabs(str_path):
        return str_path

    if not base:
        raise ValueError('cannot compute abspath without a base')

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.abspath(os.path.join(_to_str(base), str_path))


def abs_web_path(path: str, basedir: str) -> Optional[str]:
    """Return an absolute path in a base dir and ensure the path is correct.

    Args:
        path: Path to absolutize.
        basedir: Path to base directory.

    Returns:
        Absolute path with respect to base directory.
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
        gws.log.warning(f'abs_web_path: invalid dirname in path={path!r}')
        return

    if not re.match(_fil_re, fname):
        gws.log.warning(f'abs_web_path: invalid filename in path={path!r}')
        return

    p = basedir
    if dirs:
        p += '/' + '/'.join(dirs)
    p += '/' + fname

    if not os.path.isfile(p):
        gws.log.warning(f'abs_web_path: not a file path={path!r}')
        return

    return p


def rel_path(path: _Path, base: _Path) -> str:
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

    return os.path.relpath(_to_str(path), _to_str(base))


def _to_str(p: _Path) -> str:
    return p if isinstance(p, str) else p.decode('utf8')


def _to_bytes(p: _Path) -> bytes:
    return p if isinstance(p, bytes) else p.encode('utf8')
