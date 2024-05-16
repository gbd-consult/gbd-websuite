import re

import gws

MM_PER_IN = 25.4
"""Conversion factor from inch to millimetre"""
PT_PER_IN = 72
"""Conversion factor from inch to points"""
# OGC's 1px = 0.28mm
# OGC 06-042, 7.2.4.6.9

OGC_M_PER_PX = 0.00028
"""Open Geospatial Consortium standard conversion factor pixel to metre"""
OGC_SCREEN_PPI = MM_PER_IN / (OGC_M_PER_PX * 1000)  # 90.71
"""Pixel per inch on screen using the Open Geospatial Consortium standard"""
PDF_DPI = 96
"""Dots per inch in a pdf file"""
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

    Returns:
        Resolution in pixel.
    """
    # return round(x * OGC_M_PER_PX, 4)
    return x * OGC_M_PER_PX


def res_to_scale(x: _number) -> int:
    """Converts the user's resolution to the scale.

    Args:
        x: Resolution in pixel per inch.

    Returns:
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

    Returns:
        Amount of pixels."""
    return (x * ppi) / MM_PER_IN


def to_px(xu: gws.UomValue, ppi: int) -> gws.UomValue:
    """Converts a measurement to amount of pixels.

    Args:
        xu: A measurement to convert to pixels.
        ppi: Pixels per inch.

    Returns:
        A measurement.
    """
    x, u = xu
    if u == gws.Uom.px:
        return xu
    if u == gws.Uom.mm:
        return mm_to_px(x, ppi), gws.Uom.px
    raise ValueError(f'invalid unit {u!r}')


def size_mm_to_px(xy: gws.Size, ppi: int) -> gws.Size:
    """Converts a rectangle description in millimetres to pixels.

    Args:
        xy: A rectangle measurements in mm.
        ppi: Pixels per inch.

    Returns:
        A rectangle in pixel.
    """
    x, y = xy
    return mm_to_px(x, ppi), mm_to_px(y, ppi)


def size_to_px(xyu: gws.UomSize, ppi: int) -> gws.UomSize:
    """Converts a rectangle description of any unit to pixels.

    Args:
        xyu: A rectangle measurements with its unit.
        ppi: Pixels per inch.

    Returns:
        The rectangle measurements in pixels.
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

    Returns:
        Amount of millimetres.
    """
    return (x / ppi) * MM_PER_IN


def to_mm(xu: gws.UomValue, ppi: int) -> gws.UomValue:
    """Converts a measurement of any unit to millimetres.

    Args:
        xu: A measurement to convert.
        ppi: Pixels per inch.

    Returns:
        A measurement.
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
        xy: A rectangle measurements in pixels.
        ppi: Pixel per inch

    Returns:
        The rectangle measurements in millimetres.
    """
    x, y = xy
    return px_to_mm(x, ppi), px_to_mm(y, ppi)


def size_to_mm(xyu: gws.UomSize, ppi: int) -> gws.UomSize:
    """Converts a rectangle description of any unit to millimetres.

    Args:
        xyu: A rectangle measurements with its unit.
        ppi: Pixels per inch.

    Returns:
        The rectangle measurements in millimetres.
    Raises:
        ``ValueError``: if the unit is invalid.
    """
    x, y, u = xyu
    if u == gws.Uom.mm:
        return xyu
    if u == gws.Uom.px:
        return px_to_mm(x, ppi), px_to_mm(y, ppi), gws.Uom.mm
    raise ValueError(f'invalid unit {u!r}')


def to_str(xu: gws.UomValue) -> str:
    """Converts a to a string.

    Args:
        xu: A measurement to convert.

    Returns:
        The input tuple as a string, like '5mm'."""
    x, u = xu
    sx = str(int(x)) if (x % 1 == 0) else str(x)
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


def parse(s: str | int | float, default_unit: gws.Uom = None) -> gws.UomValue:
    """Parse a measurement in the string or numeric form.

    Args:
        s: A measurement to parse.
        default_unit: Default unit.

    Returns:
        A measurement.

    Raises:
         ``ValueError``: if the unit is missing, if the formatting is wrong or if the unit is invalid.
    """
    if isinstance(s, (int, float)):
        if not default_unit:
            raise ValueError(f'missing unit: {s!r}')
        return s, default_unit

    s = gws.u.to_str(s).strip()
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


_DURATION_UNITS = {
    'w': 3600 * 24 * 7,
    'd': 3600 * 24,
    'h': 3600,
    'm': 60,
    's': 1,
}


def parse_duration(s: str) -> int:
    """Converts weeks, days, hours or minutes to seconds.

    Args:
        s: Time of duration.

    Returns:
        Input as seconds.
    Raises:
        ``ValueError``: if the duration is invalid.
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
        if p is None or v not in _DURATION_UNITS:
            raise ValueError('invalid duration', s)
        r += p * _DURATION_UNITS[v]
        p = None

    if p:
        r += p

    return r
