"""Logging facility."""

import os
import sys
import traceback


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
    global _current_level
    if isinstance(level, int) or level.isdigit():
        _current_level = int(level)
    else:
        _current_level = getattr(Level, level.upper())


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
    ls = exception_backtrace(exc)
    _raw(Level.ERROR, msg or ls[0], args, kwargs)
    for s in ls[1:]:
        _raw(Level.ERROR, 'EXCEPTION :: ' + s)


def if_debug(fn, *args):
    """If debugging, apply the function to args and log the result."""

    if Level.DEBUG < _current_level:
        return
    try:
        msg = fn(*args)
    except Exception as exc:
        msg = repr(exc)
    _raw(Level.DEBUG, msg)


def exception_backtrace(exc: BaseException | None) -> list:
    """Exception backtrace as a list of strings."""

    head = _name(exc)
    messages = []

    lines = []
    pfx = ''

    while exc:
        subhead = _name(exc)
        msg = _message(exc)
        if msg:
            subhead += ': ' + msg
            messages.append(msg)
        if pfx:
            subhead = pfx + ' ' + subhead

        lines.append(subhead)

        for f in traceback.extract_tb(exc.__traceback__, limit=100):
            lines.append(f'    in {f[2]} ({f[0]}:{f[1]})')

        if exc.__cause__:
            exc = exc.__cause__
            pfx = 'caused by'
        elif exc.__context__:
            exc = exc.__context__
            pfx = 'during handling of'
        else:
            break

    if messages:
        head += ': ' + messages[0]
    if len(lines) > 1:
        head += ' ' + lines[1].strip()

    lines.insert(0, head)
    return lines


##

def _name(exc):
    typ = type(exc) or Exception
    # if typ == Error:
    #     return 'Error'
    name = getattr(typ, '__name__', '')
    mod = getattr(typ, '__module__', '')
    if mod in {'exceptions', 'builtins'}:
        return name
    return mod + '.' + name


def _message(exc):
    try:
        return repr(exc.args[0])
    except:
        return ''


##


_current_level = Level.INFO

_out_stream = sys.stdout

_PREFIX = {
    Level.CRITICAL: 'CRITICAL',
    Level.ERROR: 'ERROR',
    Level.WARNING: 'WARNING',
    Level.INFO: 'INFO',
    Level.DEBUG: 'DEBUG',
}


def _raw(level, msg, args=None, kwargs=None):
    if level < _current_level:
        return

    if args:
        if len(args) == 1 and args[0] and isinstance(args[0], dict):
            args = args[0]
        msg = msg % args

    pid = os.getpid()
    loc = ' '
    if _current_level <= Level.DEBUG:
        stacklevel = kwargs.get('stacklevel', 1) if kwargs else 1
        loc = ' ' + _location(2 + stacklevel) + ' '
    pfx = '[' + str(pid) + ']' + loc + _PREFIX[level] + ' :: '

    try:
        _out_stream.write(f'{pfx}{msg}\n')
    except UnicodeEncodeError:
        _out_stream.write(f'{pfx}{msg!r}\n')

    _out_stream.flush()


def _location(stacklevel):
    frames = traceback.extract_stack()
    for fname, line, func, text in reversed(frames):
        if stacklevel == 0:
            return f'{fname}:{line}'
        stacklevel -= 1
    return '???'
