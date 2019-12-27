import re
import gws

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


def mm2px_2(xy, ppi):
    return mm2px(xy[0], ppi), mm2px(xy[1], ppi)


def px2mm_2(xy, ppi):
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


def parse(s, unit=None):
    if isinstance(s, (int, float)):
        if not unit:
            raise ValueError('parse_unit: unit required', s)
        n = float(s)
        u = gws.as_str(unit).lower()
    else:
        s = gws.as_str(s).strip()
        m = _unit_re.match(s)
        if not m:
            raise ValueError('parse_unit: not a number', s)
        n = float(m.group('number'))
        u = (m.group('rest').strip() or unit).lower()

    if u == 'm':
        u = 'mm'
        n *= 1000

    elif u == 'cm':
        u = 'mm'
        n *= 100

    if u not in ('mm', 'in', 'px'):
        raise ValueError('parse_unit: invalid unit', s)

    return n, u


