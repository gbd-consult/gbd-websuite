"""App Error object"""

import traceback


class Error(Exception):
    def __repr__(self):
        return to_string_list(self)[0]


class ConfigurationError(Error):
    pass


def to_string_list(exc):
    head = _name(exc)
    msg = _message(exc, chain=True)
    if msg:
        head += ': ' + msg

    lines = []
    pfx = ''

    while exc:
        subhead = _name(exc)
        msg = _message(exc, chain=False)
        if msg:
            subhead += ': ' + msg
        if pfx:
            subhead = pfx + ' ' + subhead

        lines.append(subhead)

        for f in traceback.extract_tb(exc.__traceback__, limit=20):
            lines.append(f'    in {f[2]} ({f[0]}:{f[1]})')

        if exc.__cause__:
            exc = exc.__cause__
            pfx = 'caused by'
        elif exc.__context__:
            exc = exc.__context__
            pfx = 'during handling of'
        else:
            break

    if lines:
        head += ' ' + lines[1].strip()

    return [head] + lines


def _name(exc):
    t = type(exc) or Exception
    if t == Error:
        return 'gws.Error'
    if t == ConfigurationError:
        return 'gws.ConfigurationError'
    name = getattr(t, '__name__', '')
    mod = getattr(t, '__module__', '')
    if mod in {'exceptions', 'builtins'}:
        return name
    return mod + '.' + name


def _message(exc, chain=False):
    while exc:
        try:
            return repr(exc.args[0])
        except:
            pass
        if not chain:
            break
        exc = exc.__cause__
    return ''
