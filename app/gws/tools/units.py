import re
import gws

import gws.types as t

# OGC's 1px = 0.28mm
# https://portal.opengeospatial.org/files/?artifact_id=14416 page 27

OGC_M_PER_PX = 0.00028
OGC_SCREEN_PPI = 25.4 / OGC_M_PER_PX / 1000

PDF_DPI = 96

MM_PER_IN = 25.4
PT_PER_IN = 72


def scale2res(x):
    return x * OGC_M_PER_PX


def res2scale(x):
    return x / OGC_M_PER_PX


def mm2in(x):
    return x / MM_PER_IN


def m2in(x):
    return (x / MM_PER_IN) * 1000


def in2mm(x):
    return x * MM_PER_IN


def in2m(x):
    return (x * MM_PER_IN) / 1000


def in2px(x, ppi):
    return x * ppi


def mm2px(x, ppi):
    return int((x * ppi) / MM_PER_IN)


def px2mm(x, ppi):
    return int((x / ppi) * MM_PER_IN)


def point_mm2px(xy, ppi):
    return mm2px(xy[0], ppi), mm2px(xy[1], ppi)


def point_px2mm(xy, ppi):
    return px2mm(xy[0], ppi), px2mm(xy[1], ppi)


def mm2pt(x):
    return (x / MM_PER_IN) * PT_PER_IN


def pt2mm(x):
    return (x / PT_PER_IN) * MM_PER_IN


_unit_re = re.compile(r'''(?x)
    ^
        (?P<number>
            -?
            (\d+ (\.\d*)? )
            |
            (\.\d+)
        )
        (?P<rest> .*)
    $
''')

_METRIC = {
    'mm': 1,
    'cm': 10,
    'm': 1e3,
    'km': 1e6,
}


def parse(s: str, units: t.List = [], default=None) -> t.Measurement:
    if isinstance(s, (int, float)):
        if not default:
            raise ValueError(f'invalid unit value: {s!r}')
        return s, default

    s = gws.as_str(s).strip()
    m = _unit_re.match(s)
    if not m:
        raise ValueError(f'invalid unit value: {s!r}')

    n = float(m.group('number'))
    u = m.group('rest').strip().lower()

    if not units and default:
        units = [default]

    if not u:
        if not default:
            raise ValueError(f'invalid unit value: {s!r}')
        return n, default

    if u in units:
        return n, u

    # e.g. 1cm given, but only mm allowed

    if u in _METRIC:
        mm = n * _METRIC[u]
        for unit, f in _METRIC.items():
            if unit in units:
                return mm / f, unit

    # @TODO: in, ft etc

    raise ValueError(f'invalid unit value: {s!r}')


_durations = {
    'w': 3600 * 24 * 7,
    'd': 3600 * 24,
    'h': 3600,
    'm': 60,
    's': 1,
}


def parse_duration(s):
    if isinstance(s, int):
        return s

    p = None
    r = 0

    for n, v in re.findall(r'(\d+)|(\D+)', str(s).strip()):
        if n:
            p = int(n)
            continue
        v = v.strip()
        if p is None or v not in _durations:
            raise ValueError('invalid duration', s)
        r += p * _durations[v]
        p = None

    if p:
        r += p

    return r
