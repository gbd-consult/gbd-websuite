import re

import gws
import gws.types as t

MM_PER_IN = 25.4
PT_PER_IN = 72

MM = 'mm'
PX = 'px'
FT = 'ft'
IN = 'in'
PT = 'pt'
M = 'm'
KM = 'km'

# OGC's 1px = 0.28mm
# OGC 06-042, 7.2.4.6.9

OGC_M_PER_PX = 0.00028
OGC_SCREEN_PPI = MM_PER_IN / (OGC_M_PER_PX * 1000)  # 90.71

PDF_DPI = 96

_number = t.Union[int, float]


def scale_to_res(x: _number) -> float:
    # return round(x * OGC_M_PER_PX, 4)
    return x * OGC_M_PER_PX


def res_to_scale(x: _number) -> int:
    return int(x / OGC_M_PER_PX)


# @TODO imperial units not used yet
#
# def mm_to_in(x: _number) -> float:
#     return x / MM_PER_IN
#
#
# def m_to_in(x: _number) -> float:
#     return (x / MM_PER_IN) * 1000
#
#
# def in_to_mm(x: _number) -> float:
#     return x * MM_PER_IN
#
#
# def in_to_m(x: _number) -> float:
#     return (x * MM_PER_IN) / 1000
#
#
# def in_to_px(x, ppi):
#     return x * ppi
#
#
# def mm_to_pt(x: _number) -> float:
#     return (x / MM_PER_IN) * PT_PER_IN
#
#
# def pt_to_mm(x: _number) -> float:
#     return (x / PT_PER_IN) * MM_PER_IN
#

##

def mm_to_px(x: _number, ppi: int) -> float:
    return (x * ppi) / MM_PER_IN


def to_px(xu: gws.Measurement, ppi: int) -> gws.Measurement:
    x, u = xu
    if u == PX:
        return xu
    if u == MM:
        return mm_to_px(x, ppi), PX
    raise ValueError(f'invalid unit {u!r}')


def size_mm_to_px(xy: gws.Size, ppi: int) -> gws.Size:
    x, y = xy
    return mm_to_px(x, ppi), mm_to_px(y, ppi)


def msize_to_px(xyu: gws.MSize, ppi: int) -> gws.MSize:
    x, y, u = xyu
    if u == PX:
        return xyu
    if u == MM:
        return mm_to_px(x, ppi), mm_to_px(y, ppi), PX
    raise ValueError(f'invalid unit {u!r}')


##

def px_to_mm(x: _number, ppi: int) -> float:
    return (x / ppi) * MM_PER_IN


def to_mm(xu: gws.Measurement, ppi: int) -> gws.Measurement:
    x, u = xu
    if u == MM:
        return xu
    if u == PX:
        return px_to_mm(x, ppi), MM
    raise ValueError(f'invalid unit {u!r}')


def size_px_to_mm(xy: gws.Size, ppi: int) -> gws.Size:
    x, y = xy
    return px_to_mm(x, ppi), px_to_mm(y, ppi)


def msize_to_mm(xyu: gws.MSize, ppi: int) -> gws.MSize:
    x, y, u = xyu
    if u == MM:
        return xyu
    if u == PX:
        return px_to_mm(x, ppi), px_to_mm(y, ppi), MM
    raise ValueError(f'invalid unit {u!r}')


##


_unit_re = re.compile(r'''(?x)
    ^
        (?P<number>
            -?
            (\d+ (\.\d*)? )
            |
            (\.\d+)
        )
        (?P<unit> \s* [a-zA-Z]*)
    $
''')

_METRIC = {
    'mm': 1,
    'cm': 10,
    'm': 1e3,
    'km': 1e6,
}


def parse(s: str, default=None) -> gws.Measurement:
    if isinstance(s, (int, float)):
        if not default:
            raise ValueError(f'invalid unit {s!r}')
        return s, default

    s = gws.to_str(s).strip()
    m = _unit_re.match(s)
    if not m:
        raise ValueError(f'invalid unit {s!r}')

    n = float(m.group('number'))
    u = m.group('unit').strip().lower()

    if not u:
        if not default:
            raise ValueError(f'invalid unit {s!r}')
        return n, default

    return n, u


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
