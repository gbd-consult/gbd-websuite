import contextlib
import os
import getpass

import argh

import gws.config.loader
import gws.config.error
import gws.gis.mpx.config
import gws.tools.json2 as json2
import gws.tools.misc as misc


def find_commands():
    cs = {}

    for path in misc.find_files(gws.APP_DIR, r'\bcli\.py$'):
        name = path[len(gws.APP_DIR) + 1:-3].replace('/', '.')
        mod = misc.load_source(path, name)
        fns = [
            v
            for k, v in vars(mod).items()
            if not k.startswith('_') and callable(v) and v is not argh.arg
        ]
        if mod.COMMAND not in cs:
            cs[mod.COMMAND] = []
        cs[mod.COMMAND].extend(fns)

    return cs


@contextlib.contextmanager
def pretty_errors():
    ln = '-' * 40
    try:
        yield
    except gws.config.error.ParseError as e:
        _perr(ln)
        _perr('CONFIGURATION ERROR')
        _perr(e.args[0])
        _perr('path : %s' % e.args[1])
        _perr('key  : %s' % e.args[2])
        _perr('value: %s' % e.args[3])
        _perr(ln)
        raise
    except gws.config.error.LoadError as e:
        _perr(ln)
        _perr('CONFIGURATION ERROR')
        _perr_error(e)
        _perr(ln)
        raise
    except gws.config.error.MapproxyConfigError as e:
        _perr(ln)
        _perr('MAPPROXY CONFIGURATION ERROR')
        _perr_error(e)
        _perr(ln)
        raise
    except Exception as e:
        _perr(ln)
        _perr('SYSTEM ERROR')
        _perr_error(e)
        _perr(ln)
        raise


def _perr(s):
    gws.log.error(s)


def _perr_error(e):
    for arg in e.args:
        for s in gws.lines(str(arg)):
            _perr(s)


def _to_dict(x):
    try:
        v = vars(x)
        cls = x.__class__.__name__
        v['$'] = repr(x)
        return v
    except TypeError:
        return {}


def _prop_list(x, res, pfx, seen):
    if x is None or isinstance(x, (int, float, bool, str, bytes, bytearray)):
        res.append((pfx, x))
        return res

    if isinstance(x, dict):
        for k in sorted(x):
            _prop_list(x[k], res, pfx + (str(k),), seen)
        return res

    if isinstance(x, (list, tuple)):
        for k, v in enumerate(x):
            _prop_list(v, res, pfx + ('[%d]' % k,), seen)
        return res

    if x in seen:
        return res
    seen.append(x)

    return _prop_list(_to_dict(x), res, pfx, seen)


def _prop_tree(x, seen):
    if x is None or isinstance(x, (int, float, bool, str, bytes, bytearray)):
        return x

    if isinstance(x, dict):
        return {k: _prop_tree(v, seen) for k, v in sorted(x.items())}

    if isinstance(x, (list, tuple)):
        return [_prop_tree(v, seen) for v in x]

    if x in seen:
        return '@' + repr(x)
    seen.append(x)

    return _prop_tree(_to_dict(x), seen)


def dump_props(x, skip_empty=True):
    prev = []
    plist = _prop_list(x, [], (), [])

    for keys, val in plist:
        if skip_empty and gws.is_empty(val):
            continue

        kb = []

        i = 0
        while i < len(prev) and keys[i] == prev[i]:
            kb.append(' ' * len(keys[i]))
            i += 1
        kb += keys[i:]
        prev = keys

        print('%s=%r' % ('.'.join(kb), val))


def dump_json(x):
    t = _prop_tree(x, [])
    j = json2.to_string(t, pretty=True)
    print(j)


def database_credentials():
    if 'PGUSER' in os.environ and 'PGPASSWORD' in os.environ:
        return os.environ['PGUSER'], os.environ['PGPASSWORD']

    user = input('DB username: ')
    password = getpass.getpass('DB password: ')
    return user, password


def find_action(action_type, project_uid=None):
    gws.config.loader.load()

    app = gws.config.find_first('gws.common.application')
    action = app.find_action(action_type, project_uid)

    if not action:
        gws.log.error(f'{action_type!r} action not configured')

    return action


def text_table(data, header=None):
    """Format a list of dicts as a text-mode table."""

    data = list(data)

    if not data:
        return ''

    header = header or sorted(data[0].keys())
    widths = [len(h) for h in header]

    for d in data:
        widths = [
            max(a, b)
            for a, b in zip(
                widths,
                [len(str(d.get(h, ''))) for h in header]
            )
        ]

    def field(n, v):
        if isinstance(v, (int, float)):
            return str(v).rjust(widths[n])
        return str(v).ljust(widths[n])

    def make_row(a):
        return ' | '.join(a)

    rows = []

    rows.append(make_row(field(n, h) for n, h in enumerate(header)))
    rows.append('-' * len(rows[0]))

    for d in data:
        rows.append(make_row(field(n, d.get(h, '')) for n, h in enumerate(header)))

    return '\n'.join(rows)
