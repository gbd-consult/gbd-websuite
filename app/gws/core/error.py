"""App Error object"""

import traceback


class Error(Exception):
    def __repr__(self):
        return to_string_list(self)[0]


class ConfigurationError(Error):
    pass


class NotFoundError(Error):
    """Generic 'object not found' error."""
    pass


class ForbiddenError(Error):
    """Generic 'forbidden' error."""
    pass


class BadRequestError(Error):
    """Generic 'bad request' error."""
    pass


##


def to_string_list(exc: Exception) -> list:
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
    if lines:
        head += ' ' + lines[1].strip()

    lines.insert(0, head)
    return lines


##

def _name(exc):
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
