import re

import gws

MM_PER_IN = 25.4
"""Conversion factor from inch to millimetre"""

PT_PER_IN = 72
"""Conversion factor from inch to points"""

OGC_M_PER_PX = 0.00028
"""OGC meter per pixel (OGC 06-042, 7.2.4.6.9: 1px = 0.28mm)."""

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
    return x * (ppi / MM_PER_IN)


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
    return x * (MM_PER_IN / ppi)


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


_unit_re = re.compile(r"""(?x)
    ^
        (?P<number>
            -?
            (\d+ (\.\d*)? )
            |
            (\.\d+)
        )
        (?P<unit> \s* [a-zA-Z]*)
    $
""")


def parse(val: str | int | float | tuple | list, default_unit: gws.Uom = None) -> gws.UomValue:
    """Parse a measurement in the string or numeric form.

    Args:
        val: A measurement to parse (e.g. '5mm', 5, [5, 'mm']).
        default_unit: Default unit.

    Raises:
         ``ValueError``: if the unit is missing, if the formatting is wrong or if the unit is invalid.
    """
    if isinstance(val, (list, tuple)):
        if len(val) == 2:
            return parse(f'{val[0]}{val[1]}')
        raise ValueError(f'invalid format: {val!r}')

    if isinstance(val, (int, float)):
        if not default_unit:
            raise ValueError(f'missing unit: {val!r}')
        return val, default_unit

    val = gws.u.to_str(val).strip()
    m = _unit_re.match(val)
    if not m:
        raise ValueError(f'invalid format: {val!r}')

    n = float(m.group('number'))
    u = getattr(gws.Uom, m.group('unit').strip().lower(), None)

    if not u:
        if not default_unit:
            raise ValueError(f'invalid unit: {val!r}')
        return n, default_unit

    return n, u


def parse_point(val: str | tuple | list) -> gws.UomPoint:
    """Parse a point in the string or numeric form.

    Args:
        val: A point to parse, either a string '1mm 2mm' or a list [1, 2, 'mm'].

    Raises:
        ``ValueError``: if the point is invalid.
    """

    v = gws.u.to_list(val)

    if len(v) == 3:
        v = [f'{v[0]}{v[2]}', f'{v[1]}{v[2]}']

    if len(v) == 2:
        n1, u1 = parse(v[0])
        n2, u2 = parse(v[1])
        if u1 != u2:
            raise ValueError(f'invalid point units: {u1!r} != {u2!r}')
        return n1, n2, u1

    raise ValueError(f'invalid point: {val!r}')


def parse_extent(val: str | tuple | list) -> gws.UomExtent:
    """Parse an extent in the string or numeric form.

    Args:
        val: An extent to parse, either a string '1mm 2mm 3mm 4mm' or a list [1, 2, 3, 4, 'mm'].

    Raises:
        ``ValueError``: if the extent is invalid.
    """

    v = gws.u.to_list(val)

    if len(v) == 5:
        v = [f'{v[0]}{v[4]}', f'{v[1]}{v[2]}', f'{v[2]}{v[4]}', f'{v[3]}{v[4]}']

    if len(v) == 4:
        n1, u1 = parse(v[0])
        n2, u2 = parse(v[1])
        n3, u3 = parse(v[2])
        n4, u4 = parse(v[3])
        if u1 != u2 or u1 != u3 or u1 != u4:
            raise ValueError(f'invalid extent units: {u1!r} != {u2!r} != {u3!r} != {u4!r}')
        return n1, n2, n3, n4, u1

    raise ValueError(f'invalid extent: {val!r}')
