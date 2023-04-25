"""Debuggging tools"""

import collections
import re
import socket
import sys
import time
import traceback

from . import log

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


def _dump(x, name, depth, max_depth, all_props):
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

    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        yield pfx + ' = ' + _repr(x)
        return

    if not all_props and str(type(x)) in _noexpand:
        yield pfx + ' = ' + _repr(x)
        return

    if depth >= max_depth:
        yield pfx + '...'
        return

    if isinstance(x, dict) and x:
        yield pfx + ' = '
        for k in x:
            for s in _dump(x[k], repr(k), depth + 1, max_depth, all_props):
                yield s

    elif isinstance(x, (set, list)) and x:
        yield pfx + ' = '
        for k, v in enumerate(x):
            for s in _dump(v, str(k), depth + 1, max_depth, all_props):
                yield s

    else:
        yield pfx + ' = ' + _repr(x)

    for k in dir(x):
        try:
            v = getattr(x, k)
        except Exception as e:
            v = '?' + repr(e)
        if all_props or _should_list(k, v):
            for s in _dump(v, k, depth + 1, max_depth, all_props):
                yield s


def inspect(arg, max_depth=1, all_props=False):
    """Inspect the argument upto the given depth"""

    for s in _dump(arg, None, 0, max_depth or 1, all_props):
        yield s


def p(*args, **kwargs):
    sep = '-' * 60

    if kwargs.get('lines'):
        for arg in args:
            for s in enumerate(str(arg).split('\n'), 1):
                log.debug('%06d:%s' % s, stacklevel=2)
            log.debug(sep, stacklevel=2)
    else:
        max_depth = kwargs.get('d', 3)
        all_props = kwargs.get('all', False)
        for arg in args:
            for s in inspect(arg, max_depth=max_depth, all_props=all_props):
                log.debug(s, stacklevel=2)
            log.debug(sep, stacklevel=2)

    if kwargs.get('stack'):
        for s in traceback.format_stack()[:-1]:
            log.debug(s.replace('\n', ' '), stacklevel=2)


_timers = {}


def time_start(label):
    _timers[label] = time.time()


def time_end(label):
    if label in _timers:
        ts = time.time() - _timers.pop(label)
        log.debug('TIMER %s=%.2f', label, ts, stacklevel=2)


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
