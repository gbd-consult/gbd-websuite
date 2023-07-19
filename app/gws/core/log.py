"""Logging facility."""

import os
import sys
import traceback

from . import error as err


class Level:
    CRITICAL = 50
    ERROR = 40
    WARN = 30
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0
    ALL = 0


def set_level(level: int | str):
    global _CURLEVEL
    if isinstance(level, int) or level.isdigit():
        _CURLEVEL = int(level)
    else:
        _CURLEVEL = getattr(Level, level.upper())


def log(level: int, msg: str, *args, **kwargs):
    _raw(level, msg, args, kwargs)


def critical(msg: str, *args, **kwargs):
    _raw(Level.CRITICAL, msg, args, kwargs)


def error(msg: str, *args, **kwargs):
    _raw(Level.ERROR, msg, args, kwargs)


def warning(msg: str, *args, **kwargs):
    _raw(Level.WARNING, msg, args, kwargs)


def info(msg: str, *args, **kwargs):
    _raw(Level.INFO, msg, args, kwargs)


def debug(msg: str, *args, **kwargs):
    _raw(Level.DEBUG, msg, args, kwargs)


def exception(msg: str = '', *args, **kwargs):
    _, exc, _ = sys.exc_info()
    ls = err.to_string_list(exc)
    _raw(Level.ERROR, msg or ls[0], args, kwargs)
    for s in ls[1:]:
        _raw(Level.DEBUG, 'EXCEPTION :: ' + s)


def if_debug(fn, *args):
    """If debugging, apply the function to args and log the result."""

    if Level.DEBUG < _CURLEVEL:
        return
    try:
        msg = fn(*args)
    except Exception as exc:
        msg = repr(exc)
    _raw(Level.DEBUG, msg)


##


_CURLEVEL = Level.INFO

_OUT = sys.stdout

_PREFIX = {
    Level.CRITICAL: 'CRITICAL',
    Level.ERROR: 'ERROR',
    Level.WARNING: 'WARNING',
    Level.INFO: 'INFO',
    Level.DEBUG: 'DEBUG',
}


def _raw(level, msg, args=None, kwargs=None):
    if level < _CURLEVEL:
        return

    if args:
        if len(args) == 1 and args[0] and isinstance(args[0], dict):
            args = args[0]
        msg = msg % args

    pid = os.getpid()
    loc = ' '
    if _CURLEVEL <= Level.DEBUG:
        stacklevel = kwargs.get('stacklevel', 1) if kwargs else 1
        loc = ' ' + _location(2 + stacklevel) + ' '
    pfx = '[' + str(pid) + ']' + loc + _PREFIX[level] + ' :: '

    try:
        _OUT.write(f'{pfx}{msg}\n')
    except UnicodeEncodeError:
        _OUT.write(f'{pfx}{msg!r}\n')

    _OUT.flush()


def _location(stacklevel):
    frames = traceback.extract_stack()
    for fname, line, func, text in reversed(frames):
        if stacklevel == 0:
            return f'{fname}:{line}'
        stacklevel -= 1
    return '???'
