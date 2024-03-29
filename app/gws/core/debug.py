"""Debuggging tools"""

import re
import socket
import sys
import time
import traceback

from . import log


def inspect(arg, max_depth=1, all_props=False):
    """Inspect the argument upto the given depth"""

    yield from _dump(arg, None, 0, max_depth or 1, all_props, [])


def p(*args, lines=False, stack=False, d=3, all=False):
    """Log an inspection of the arguments"""

    sep = '-' * 60

    if lines:
        for arg in args:
            for s in enumerate(str(arg).split('\n'), 1):
                log.debug('%06d:%s' % s, stacklevel=2)
            log.debug(sep, stacklevel=2)
        return

    if stack:
        for s in traceback.format_stack()[:-1]:
            log.debug(s.replace('\n', ' '), stacklevel=2)
        return

    for arg in args:
        for s in inspect(arg, max_depth=d, all_props=all):
            log.debug(s, stacklevel=2)
        log.debug(sep, stacklevel=2)


_TIME_STACK = []


def time_start(label=None):
    _TIME_STACK.append((time.time(), label or 'default'))


def time_end():
    if _TIME_STACK:
        t2 = time.time()
        t1, label = _TIME_STACK.pop()
        log.debug(f'@PROFILE {label} :: {t2 - t1:.2f}', stacklevel=2)


def pycharm_debugger_check(path_to_pycharm_debug_egg, host, port, suspend=False):
    """Check for pycharm debugger listeniing.

    Attempt to open the debugger socket first and return that socket when pydevd asks for it.
    If there is no socket, then IDEA is not listening, return quickly.
    """

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
    except:
        sock = None

    if not sock:
        return False

    if not path_to_pycharm_debug_egg.endswith('.egg'):
        if sys.version_info >= (3, 0):
            path_to_pycharm_debug_egg += '/pycharm-debug-py3k.egg'
        else:
            path_to_pycharm_debug_egg += '/pycharm-debug.egg'

    sys.path.append(path_to_pycharm_debug_egg)
    pydevd = __import__('pydevd')

    pydevd.StartClient = lambda h, p: sock
    pydevd.start_client = lambda h, p: sock
    pydevd.settrace(host, port=port, stdoutToServer=True, stderrToServer=True, suspend=suspend)

    return True


##


_noexpand = {
    "<class 'datetime.datetime'>",
    "<class 'datetime.date'>",
    "<class 'memoryview'>",
}


def _should_list(k, v):
    if k.startswith('__'):
        return False
    if isinstance(v, type):
        return True
    if callable(v):
        return False
    return True


_MAX_REPR_LEN = 255


def _repr(x):
    s = repr(x)
    if len(s) > _MAX_REPR_LEN:
        s = s[:_MAX_REPR_LEN] + '...'
    return s


def _dump(x, name, depth, max_depth, all_props, seen):
    pfx = '    ' * depth

    if name:
        pfx += str(name) + ': '

    t = str(type(x))
    m = re.match(r"^<(type|class) '(.+?)'>", t)
    pfx += m.group(2) if m else t

    try:
        pfx += '(%d)' % len(x)
    except (TypeError, AttributeError):
        pass

    head = pfx + ' = ' + _repr(x)

    if not x or isinstance(x, (bool, int, float, str, bytes)):
        yield head
        return

    if not all_props and str(type(x)) in _noexpand:
        yield head
        return

    try:
        if x in seen:
            yield head
            return
        seen.append(x)
    except:
        pass

    if depth >= max_depth:
        yield pfx + '...'
        return

    if isinstance(x, dict):
        yield pfx + ' = '
        for k in x:
            yield from _dump(x[k], repr(k), depth + 1, max_depth, all_props, seen)
        return

    if isinstance(x, (set, list, tuple)):
        yield pfx + ' = '
        for k, v in enumerate(x):
            yield from _dump(v, str(k), depth + 1, max_depth, all_props, seen)
        return

    yield head

    for k in dir(x):
        try:
            v = getattr(x, k)
        except Exception as e:
            v = '?' + repr(e)
        if all_props or _should_list(k, v):
            yield from _dump(v, k, depth + 1, max_depth, all_props, seen)
