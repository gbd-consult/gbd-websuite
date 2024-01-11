import re

import gws
import gws.types as t

MM_PER_IN = 25.4
PT_PER_IN = 72

# OGC's 1px = 0.28mm
# OGC 06-042, 7.2.4.6.9

OGC_M_PER_PX = 0.00028
OGC_SCREEN_PPI = MM_PER_IN / (OGC_M_PER_PX * 1000)  # 90.71

PDF_DPI = 96

# 1 centimeter precision

DEFAULT_PRECISION = {
    gws.Uom.deg: 7,
    gws.Uom.m: 2,
}

_number = int | float


def scale_to_res(x: _number) -> float:
    """Converts the scale to the user's resolution.

    Args:
        x: Scale.

    Return:
        Resolution in pixel.
    """
    # return round(x * OGC_M_PER_PX, 4)
    return x * OGC_M_PER_PX


def res_to_scale(x: _number) -> int:
    """Converts the user's resolution to the scale.

    Args:
        x: Resolution in pixel per inch.

    Return:
        Scale.
    """
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
    """Converts millimetres to pixel.

    Args:
        x: Millimetres.
        ppi: Pixels per inch.

    Return:
        Amount of pixels without unit."""
    return (x * ppi) / MM_PER_IN


def to_px(xu: gws.Measurement, ppi: int) -> gws.Measurement:
    """Converts the tuple measurement to amount of pixels.

    Args:
        xu: ``(value, unit)``.
        ppi: Pixels per inch.

    Return:
        ``(amount of pixels, 'px')``.
    """
    x, u = xu
    if u == gws.Uom.px:
        return xu
    if u == gws.Uom.mm:
        return mm_to_px(x, ppi), gws.Uom.px
    raise ValueError(f'invalid unit {u!r}')


def size_mm_to_px(xy: gws.Size, ppi: int) -> gws.Size:
    """Converts rectangle description in millimetres to pixels.

    Args:
        xy: ``(width, height)``.
        ppi: Pixels per inch.

    Return:
        ``(width in pixels, height in pixels)``.
    """
    x, y = xy
    return mm_to_px(x, ppi), mm_to_px(y, ppi)


def msize_to_px(xyu: gws.MSize, ppi: int) -> gws.MSize:
    """Converts rectangle description of any unit to pixels.

    Args:
        xyu: ``(width, height, unit)``.
        ppi: Pixels per inch.

    Return:
        ``(converted width, converted height, 'px')``.
    """
    x, y, u = xyu
    if u == gws.Uom.px:
        return xyu
    if u == gws.Uom.mm:
        return mm_to_px(x, ppi), mm_to_px(y, ppi), gws.Uom.px
    raise ValueError(f'invalid unit {u!r}')


##

def px_to_mm(x: _number, ppi: int) -> float:
    """Converts pixel to millimetres.

    Args:
        x: Amount of pixels.
        ppi: Pixel per inch.

    Return:
        Amount of millimetres.
    """
    return (x / ppi) * MM_PER_IN


def to_mm(xu: gws.Measurement, ppi: int) -> gws.Measurement:
    """Converts the tuple measurement of any unit to millimetres.

    Args:
        xu: ``(value, unit)``.
        ppi: Pixels per inch.

    Return:
        ``(amount of millimetres, 'mm')``.
    """
    x, u = xu
    if u == gws.Uom.mm:
        return xu
    if u == gws.Uom.px:
        return px_to_mm(x, ppi), gws.Uom.mm
    raise ValueError(f'invalid unit {u!r}')


def size_px_to_mm(xy: gws.Size, ppi: int) -> gws.Size:
    """Converts a rectangle description in pixel to millimetres.

    Args:
        xy: ``(width in pixel, height in pixel)``.
        ppi: Pixel per inch

    Return:
        ``(width in millimetres, height in millimetres)``.
    """
    x, y = xy
    return px_to_mm(x, ppi), px_to_mm(y, ppi)


def msize_to_mm(xyu: gws.MSize, ppi: int) -> gws.MSize:
    """Converts the rectangle description of any unit to millimetres.

    Args:
        xyu: ``(width, height, unit)``.
        ppi: Pixels per inch.

    Return:
        ``(converted width, converted height, 'mm')``.
    """
    x, y, u = xyu
    if u == gws.Uom.mm:
        return xyu
    if u == gws.Uom.px:
        return px_to_mm(x, ppi), px_to_mm(y, ppi), gws.Uom.mm
    raise ValueError(f'invalid unit {u!r}')


##


def to_str(xu: gws.Measurement) -> str:
    """Converts the tuple measurement to a string.

    Args:
        xu: ``(value, unit)``.

    Return:
        The input tuple as a string, like '5mm'."""
    x, u = xu
    sx = str(int(x)) if x.is_integer() else str(x)
    return sx + str(u)


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


def parse(s: str, default_unit=None) -> gws.Measurement:
    """Checks if a measurement fits the format ``'value unit'``.

    Args:
        s: Measurement to check.

    Return:
        ``(value, 'unit')``.
        Raises an Error if the unit is missing, if the formatting is wrong or if the unit is invalid.
    """
    if isinstance(s, (int, float)):
        if not default_unit:
            raise ValueError(f'missing unit: {s!r}')
        return s, default_unit

    s = gws.to_str(s).strip()
    m = _unit_re.match(s)
    if not m:
        raise ValueError(f'invalid format: {s!r}')

    n = float(m.group('number'))
    u = getattr(gws.Uom, m.group('unit').strip().lower(), None)

    if not u:
        if not default_unit:
            raise ValueError(f'invalid unit: {s!r}')
        return n, default_unit

    return n, u


_durations = {
    'w': 3600 * 24 * 7,
    'd': 3600 * 24,
    'h': 3600,
    'm': 60,
    's': 1,
}


def parse_duration(s: str) -> int:
    """converts weeks, days, hours or minutes to seconds.

    Args:
        s: Time of duration.

    Return:
        Input as seconds. Raises an error if the duration is invalid.
    """
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
