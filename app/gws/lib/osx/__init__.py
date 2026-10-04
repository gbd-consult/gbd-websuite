"""Operating system and shell utilities.

This package wraps common operating system tasks used throughout GWS:

- running external commands (``run``, ``run_nowait``),
- file system operations (``unlink``, ``rename``, ``copy``, ``mkdir``, ``rmdir``, ``touch``, ``chown``),
- file information (``file_mtime``, ``file_age``, ``file_size``, ``file_checksum``),
- searching directories (``find_files``, ``find_directories``),
- path manipulation (``parse_path``, ``abs_path``, ``rel_path``, ``abs_web_path``),
- processes and users (``kill_pid``, ``running_pids``, ``process_rss_size``, ``user_info``).

Most file functions accept paths as ``str`` or ``bytes``. Functions that query
files return a sentinel value (``-1`` or an empty string) instead of raising
when the file cannot be accessed.

Example::

    import gws.lib.osx

    out = gws.lib.osx.run(['gdalinfo', '--version'])
    gws.lib.osx.mkdir('/tmp/gws/data')
    for path in gws.lib.osx.find_files('/data/projects', ext='json'):
        print(path, gws.lib.osx.file_size(path))
"""

from typing import Optional

import grp
import hashlib
import os
import pwd
import re
import shlex
import shutil
import signal
import subprocess
import time

import psutil

import gws


class Error(gws.Error):
    """Generic error raised by OS utilities."""

    pass


class TimeoutError(Error):
    """Raised when an external command times out."""

    pass


_Path = str | bytes


def getenv(key: str, default: str = None) -> Optional[str]:
    """Return the value of an environment variable.

    Args:
        key: Variable name.
        default: Value to return if the variable is not set.

    Returns:
        The variable value, or ``default`` if the variable is not set.
    """
    return os.getenv(key, default)


def run_nowait(cmd: str | list, **kwargs) -> subprocess.Popen:
    """Start a process and return immediately, without waiting for it to finish.

    By default, the process inherits stdin, stdout and stderr, and the command is not run in a shell.

    Args:
        cmd: Command to run, as a string or a list of arguments.
        kwargs: Arguments to pass to ``subprocess.Popen``.

    Returns:
        The ``subprocess.Popen`` object of the started process.
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
    """Run an external command and wait for it to finish.

    A string command is split into arguments with ``shlex.split``. The command is not run in a shell,
    and stderr is merged into stdout.

    Args:
        cmd: Command to run, as a string or a list of arguments.
        input: Data to send to the command's stdin.
        echo: Let the output go to the console instead of capturing it.
        strict: Raise an error on a non-zero exit code.
        timeout: Timeout in seconds.
        kwargs: Arguments to pass to ``subprocess.Popen``.

    Returns:
        The captured command output, or an empty string if the output was not captured.

    Raises:
        ``TimeoutError``: If the command times out.
        ``Error``: If the command cannot be run, or exits with a non-zero code and ``strict`` is true.
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
    """Delete a file.

    Directories and non-existing paths are ignored.

    Args:
        path: File path.

    Returns:
        ``True`` on success or if there was nothing to delete, ``False`` if an OS error occurred.
    """
    try:
        if os.path.isfile(path):
            os.unlink(path)
        return True
    except OSError as exc:
        gws.log.debug(f'OSError: unlink: {exc}')
        return False


def rename(src: _Path, dst: _Path):
    """Move or rename a file or directory.

    Args:
        src: Source path.
        dst: Destination path.
    """

    shutil.move(_to_str(src), _to_str(dst))


def chown(path: _Path, user: int = None, group: int = None):
    """Change the owner and group of a path.

    Args:
        path: File path.
        user: User ID, defaults to ``gws.c.UID``.
        group: Group ID, defaults to ``gws.c.GID``.
    """
    os.chown(path, user or gws.c.UID, group or gws.c.GID)


def copy(src: _Path, dst: _Path, user: int = None, group: int = None):
    """Copy a file and set the owner of the copy.

    Args:
        src: Source path.
        dst: Destination path.
        user: User ID of the copy, defaults to ``gws.c.UID``.
        group: Group ID of the copy, defaults to ``gws.c.GID``.
    """
    shutil.copyfile(src, dst)
    os.chown(dst, user or gws.c.UID, group or gws.c.GID)


def mkdir(path: _Path, mode: int = 0o755, user: int = None, group: int = None):
    """Create a directory, including missing parent directories.

    Does nothing if the directory already exists.

    Args:
        path: Path to a directory.
        mode: Directory creation mode.
        user: Directory user. Currently not used.
        group: Directory group. Currently not used.
    """

    os.makedirs(path, mode, exist_ok=True)


def rmdir(path: _Path) -> bool:
    """Remove a directory or a directory tree.

    Args:
        path: Path to a directory. Can be non-empty.

    Returns:
        ``True`` if the directory was removed, ``False`` if it does not exist or an OS error occurred.
    """

    if not os.path.isdir(path):
        return False
    try:
        shutil.rmtree(path)
        return True
    except OSError as exc:
        gws.log.warning(f'OSError: rmdir: {exc}')
        return False


def touch(path: _Path):
    """Set the access and modification times of a file to the current time.

    If the file does not exist, it is created.

    Args:
        path: File path.
    """
    with open(path, 'a'):
        os.utime(path, None)


def file_mtime(path: _Path) -> float:
    """Return the modification time of a path.

    Args:
        path: File or directory path.

    Returns:
        Modification time in seconds since the epoch, or ``-1`` if the path cannot be accessed.
    """
    try:
        return os.stat(path).st_mtime
    except OSError:
        return -1


def file_age(path: _Path) -> int:
    """Return the number of seconds since a path was last modified.

    Args:
        path: File path.

    Returns:
        Age in seconds, or ``-1`` if the path cannot be accessed.
    """
    try:
        return int(time.time() - os.stat(path).st_mtime)
    except OSError:
        return -1


def file_size(path: _Path) -> int:
    """Return the size of a file.

    Args:
        path: File path.

    Returns:
        Size in bytes, or ``-1`` if the path cannot be accessed.
    """
    try:
        return os.stat(path).st_size
    except OSError:
        return -1


def file_checksum(path: _Path) -> str:
    """Return the SHA-256 checksum of a file.

    Args:
        path: File path.

    Returns:
        Hex digest of the file content, or an empty string if the file cannot be read.
    """
    try:
        with open(path, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
    except OSError as exc:
        return ''


def kill_pid(pid: int, sig_name='TERM') -> bool:
    """Send a signal to a process.

    Args:
        pid: Process ID.
        sig_name: Signal name, with or without the ``SIG`` prefix, e.g. ``TERM`` or ``SIGKILL``.

    Returns:
        ``True`` if the signal was sent or the process does not exist, ``False`` if the signal could not be sent.
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
    """Return all running processes.

    Returns:
        A dict mapping process IDs to process names.
    """
    d = {}
    for p in psutil.process_iter():
        d[p.pid] = p.name()
    return d


def process_rss_size(unit: str = 'm') -> float:
    """Return the Resident Set Size of the current process.

    Args:
        unit: ``k`` (kilobytes), ``m`` (megabytes) or ``g`` (gigabytes). Any other value returns bytes.

    Returns:
        The Resident Set Size in the given unit.
    """
    n = psutil.Process().memory_info().rss
    if unit == 'k':
        return n / 1e3
    if unit == 'm':
        return n / 1e6
    if unit == 'g':
        return n / 1e9
    return n


def user_info(uid=None, gid=None) -> dict:
    """Return user and group information.

    Args:
        uid: User ID. Defaults to the user ID of the current process.
        gid: Group ID. Defaults to the user's primary group ID.

    Returns:
        A dict with the keys ``pw_name``, ``pw_uid``, ``pw_gid``, ``pw_dir``, ``pw_shell``
        (from ``struct_passwd``) and ``gr_name``, ``gr_gid`` (from ``struct_group``).
    """

    uid = uid or os.getuid()
    u = pwd.getpwuid(uid)

    r = dict(
        pw_name=u.pw_name,
        pw_uid=u.pw_uid,
        pw_gid=u.pw_gid,
        pw_dir=u.pw_dir,
        pw_shell=u.pw_shell,
    )

    gid = gid or r['pw_gid']
    g = grp.getgrgid(gid)

    r['gr_name'] = g.gr_name
    r['gr_gid'] = g.gr_gid

    return r


def find_entries(dirname: _Path, deep: bool = True):
    """Find entries in a directory, skipping hidden ones.

    Args:
        dirname: Path to a directory.
        deep: If true, also search subdirectories recursively.

    Yields:
        ``os.DirEntry`` objects for files and directories whose names do not start with a dot.
    """
    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue
        yield de
        if de.is_dir() and deep:
            yield from find_entries(de.path, deep=deep)


def find_files(dirname: _Path, pattern=None, ext=None, deep: bool = True):
    """Find files in a directory, skipping hidden ones.

    Args:
        dirname: Path to a directory.
        pattern: Regular expression to search for in the file path.
        ext: Extension or a list of extensions to match. Used only if ``pattern`` is not given.
        deep: If true, also search subdirectories recursively.

    Yields:
        Paths of matching files.
    """
    if not pattern and ext:
        if isinstance(ext, (list, tuple)):
            ext = '|'.join(ext)
        pattern = '\\.(' + ext + ')$'

    for de in find_entries(dirname, deep=deep):
        if de.is_file() and (pattern is None or re.search(pattern, de.path)):
            yield de.path


def find_directories(dirname: _Path, pattern=None, deep: bool = True):
    """Find directories in a directory, skipping hidden ones.

    Args:
        dirname: Path to a directory.
        pattern: Regular expression to search for in the directory path.
        deep: If true, also search subdirectories recursively.

    Yields:
        Paths of matching directories.
    """

    for de in find_entries(dirname, deep=deep):
        if de.is_dir() and (pattern is None or re.search(pattern, de.path)):
            yield de.path


class ParsePathResult(gws.Data):
    """Components of a file path, as returned by ``parse_path``."""

    path: str
    """The full path."""
    dirname: str
    """The directory part."""
    filename: str
    """The file name, including the extension."""
    stem: str
    """The file name up to the first dot."""
    extension: str
    """The file name after the first dot, without the dot."""


def parse_path(path: _Path) -> ParsePathResult:
    """Split a file path into its components.

    The extension is everything after the first dot in the file name, so ``a.tar.gz``
    has the stem ``a`` and the extension ``tar.gz``. A file name starting with a dot has no extension.

    Args:
        path: File path.

    Returns:
        The path components.
    """

    str_path = _to_str(path)
    sp = os.path.split(str_path)

    pp = ParsePathResult(
        path=str_path,
        dirname='',
        filename='',
        stem='',
        extension='',
    )
    pp.dirname = sp[0]
    pp.filename = sp[1]

    if pp.filename.startswith('.'):
        pp.stem = pp.filename
    else:
        pp.stem, _, pp.extension = pp.filename.partition('.')

    return pp


def file_name(path: _Path) -> str:
    """Return the file name part of a path.

    Args:
        path: File path.

    Returns:
        The last component of the path.
    """

    sp = os.path.split(_to_str(path))
    return sp[1]


def is_abs_path(path: _Path) -> bool:
    """Check if a path is absolute.

    Args:
        path: File path.

    Returns:
        ``True`` if the path is absolute.
    """
    return os.path.isabs(path)


def abs_path(path: _Path, base: _Path) -> str:
    """Make a relative path absolute with respect to a base directory or file path.

    If ``base`` is a file, its directory is used. An absolute ``path`` is only normalized.

    Args:
        path: A path.
        base: Base directory or file path.

    Returns:
        The absolute path.

    Raises:
        ``ValueError``: If ``path`` is relative and ``base`` is empty.
    """

    str_path = _to_str(path)

    if os.path.isabs(str_path):
        return os.path.normpath(str_path)

    if not base:
        raise ValueError('cannot compute abspath without a base')

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.abspath(os.path.join(_to_str(base), str_path))


def abs_web_path(path: str, basedir: str) -> Optional[str]:
    """Resolve a web path in a base directory.

    The path components must consist of letters, digits, ``_`` and ``-``,
    the file name can also contain lowercase extensions. This prevents path traversal.

    Args:
        path: Slash-separated path, as received from a web request.
        basedir: Path to the base directory.

    Returns:
        The path of an existing file in the base directory, or ``None`` if the path is invalid or the file does not exist.
    """

    _dir_re = r'^[A-Za-z0-9_-]+$'
    _fil_re = r'^[A-Za-z0-9_-]+(\.[a-z0-9]+)*$'

    gws.log.debug(f'abs_web_path: trying {path!r} in {basedir!r}')

    dirs = []
    for s in path.split('/'):
        s = s.strip()
        if s:
            dirs.append(s)

    if not dirs:
        gws.log.warning(f'abs_web_path: empty path={path!r}')
        return

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
    """Make a path relative to a base directory or file path.

    If ``base`` is a file, its directory is used.

    Args:
        path: Path to make relative.
        base: Base directory or file path.

    Returns:
        The relative path.
    """

    if os.path.isfile(base):
        base = os.path.dirname(base)

    return os.path.relpath(_to_str(path), _to_str(base))


def _to_str(p: _Path) -> str:
    return p if isinstance(p, str) else bytes(p).decode('utf8')


def _to_bytes(p: _Path) -> bytes:
    return p if isinstance(p, bytes) else str(p).encode('utf8')
