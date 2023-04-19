"""App Error object"""

import sys
import traceback


def _name(exc):
    t = type(exc)
    if t and getattr(t, '__module__') == 'exceptions':
        return getattr(t, '__name__')
    if t:
        return getattr(t, '__module__') + '.' + getattr(t, '__name__')
    return '<unknown>'


def string():
    fname = line = func = ''

    _, exc, tb = sys.exc_info()
    if tb:
        fname, line, func, _ = traceback.extract_tb(tb)[0]

    return '{{{ %s in %s:%s %s()\n\n%s\n}}}' % (
        _name(exc), fname, line, func, traceback.format_exc(chain=True).strip())


class Error(Exception):
    pass



def to_string_list(exc: Exception) -> list:
    """Exception backtrace as a list of strings."""

    head = _name2(exc)
    messages = []

    lines = []
    pfx = ''

    while exc:
        subhead = _name2(exc)
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
    if lines:
        head += ' ' + lines[1].strip()

    lines.insert(0, head)
    return lines


##

def _name2(exc):
    typ = type(exc) or Exception
    if typ == Error:
        return 'Error'
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
