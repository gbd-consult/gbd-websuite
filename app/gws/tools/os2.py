"""Utilities for os/shell scripting"""

import subprocess
import os
import re
import signal
import hashlib
import psutil

import gws


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


def is_file(path):
    return os.path.isfile(path)


def is_dir(path):
    return os.path.isdir(path)


try:
    _signals = {
        'ABRT': signal.SIGABRT,
        'ALRM': signal.SIGALRM,
        'BUS': signal.SIGBUS,
        'CHLD': signal.SIGCHLD,
        'CLD': signal.SIGCLD,
        'CONT': signal.SIGCONT,
        'FPE': signal.SIGFPE,
        'HUP': signal.SIGHUP,
        'ILL': signal.SIGILL,
        'INT': signal.SIGINT,
        'IO': signal.SIGIO,
        'IOT': signal.SIGIOT,
        'KILL': signal.SIGKILL,
        'PIPE': signal.SIGPIPE,
        'POLL': signal.SIGPOLL,
        'PROF': signal.SIGPROF,
        'PWR': signal.SIGPWR,
        'QUIT': signal.SIGQUIT,
        'RTMAX': signal.SIGRTMAX,
        'RTMIN': signal.SIGRTMIN,
        'SEGV': signal.SIGSEGV,
        'STOP': signal.SIGSTOP,
        'SYS': signal.SIGSYS,
        'TERM': signal.SIGTERM,
        'TRAP': signal.SIGTRAP,
        'TSTP': signal.SIGTSTP,
        'TTIN': signal.SIGTTIN,
        'TTOU': signal.SIGTTOU,
        'URG': signal.SIGURG,
        'USR1': signal.SIGUSR1,
        'USR2': signal.SIGUSR2,
        'VTALRM': signal.SIGVTALRM,
        'WINCH': signal.SIGWINCH,
        'XCPU': signal.SIGXCPU,
        'XFSZ': signal.SIGXFSZ,
    }
except:
    _signals = {}


def kill_pid(pid, sig_name='TERM'):
    try:
        psutil.Process(pid).send_signal(_signals[sig_name])
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


def find_files(dirname, pattern=None, ext=None):
    if not pattern and ext:
        if isinstance(ext, (list, tuple)):
            ext = '|'.join(ext)
        pattern = '\\.(' + ext + ')$'

    for fname in os.listdir(dirname):
        if fname.startswith('.'):
            continue

        path = os.path.join(dirname, fname)

        if os.path.isdir(path):
            yield from find_files(path, pattern)
            continue

        if pattern is None or re.search(pattern, path):
            yield path


def parse_path(path):
    """Parse a path into a dict(path,dirname,filename,name,extension)"""

    d = {'path': path}

    d['dirname'], d['filename'] = os.path.split(path)
    if d['filename'].startswith('.'):
        d['name'], d['extension'] = d['filename'], ''
    else:
        d['name'], _, d['extension'] = d['filename'].partition('.')

    return d


def abs_path(path, basedir):
    """Absolutize a relative path with respect to a base dir."""

    p = path.strip('/')

    if p.startswith('.') or '/.' in p:
        return None

    p = os.path.abspath(os.path.join(basedir, p))

    if not p.startswith(basedir):
        return None

    return p


def rel_path(path, basedir):
    """Relativize an absolute path with respect to a base dir."""

    if not path.startswith(basedir):
        return None

    return os.path.relpath(path, basedir)


def chown(path, user=None, group=None):
    try:
        os.chown(path, user or gws.UID, group or gws.GID)
    except OSError:
        pass
