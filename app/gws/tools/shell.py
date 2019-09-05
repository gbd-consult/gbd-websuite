"""Utilities for os/shell scripting"""

import subprocess
import os
import hashlib

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


def file_checksum(path):
    try:
        with open(path, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
    except OSError:
        return '0'
