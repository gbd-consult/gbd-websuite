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


class ConfigurationError(Error):
    pass
