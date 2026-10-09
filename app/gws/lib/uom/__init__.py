"""Units of measure.

Conversions between map scales and resolutions, between millimetres and pixels,
and parsing of values with units.

Values with units are represented as tuples, see the ``gws.Uom*`` types:

- ``gws.UomValue``: ``(5, gws.Uom.mm)``,
- ``gws.UomPoint`` and ``gws.UomSize``: ``(10, 20, gws.Uom.mm)``,
- ``gws.UomExtent``: ``(0, 0, 100, 200, gws.Uom.mm)``.

Conversions between millimetres and pixels need a resolution in pixels per inch.
Scale and resolution conversions use the OGC standard pixel size of 0.28 mm.

Example::

    import gws.lib.uom

    v = gws.lib.uom.parse('5mm')  # (5.0, gws.Uom.mm)
    gws.lib.uom.to_px(v, 96)  # (18.89..., gws.Uom.px)
    gws.lib.uom.to_str(v)  # '5mm'
    gws.lib.uom.parse_point('10mm,20mm')  # (10.0, 20.0, gws.Uom.mm)
    gws.lib.uom.res_to_scale(0.28)  # 1000
"""

import re

import gws

MM_PER_IN = 25.4
"""Conversion factor from inch to millimetre."""

PT_PER_IN = 72
"""Conversion factor from inch to points."""

OGC_M_PER_PX = 0.00028
"""OGC meter per pixel (OGC 06-042, 7.2.4.6.9: 1px = 0.28mm)."""

OGC_SCREEN_PPI = MM_PER_IN / (OGC_M_PER_PX * 1000)  # 90.71
"""Screen pixels per inch according to the OGC standard pixel size."""

PDF_DPI = 96
"""Dots per inch in a PDF file."""

# 1 centimeter precision

DEFAULT_PRECISION = {
    gws.Uom.deg: 7,
    gws.Uom.m: 2,
}

_number = int | float


def scale_to_res(x: _number) -> float:
    """Convert a scale denominator to a resolution.

    Args:
        x: Scale denominator.

    Returns:
        Resolution in metres per pixel, using the OGC pixel size.
    """
    # return round(x * OGC_M_PER_PX, 4)
    return x * OGC_M_PER_PX


def res_to_scale(x: _number) -> int:
    """Convert a resolution to a scale denominator.

    Args:
        x: Resolution in metres per pixel.

    Returns:
        Scale denominator, using the OGC pixel size.
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
    """Convert millimetres to pixels.

    Args:
        x: Millimetres.
        ppi: Pixels per inch.

    Returns:
        Number of pixels.
    """
    return x * (ppi / MM_PER_IN)


def to_px(v: gws.UomValue, ppi: int) -> gws.UomValue:
    """Convert a value with a unit to pixels.

    Args:
        v: Value in ``px`` or ``mm``.
        ppi: Pixels per inch.

    Returns:
        The value in pixels.

    Raises:
        ``ValueError``: If the unit is not ``px`` or ``mm``.
    """
    x, u = v
    if u == gws.Uom.px:
        return v
    if u == gws.Uom.mm:
        return mm_to_px(x, ppi), gws.Uom.px
    raise ValueError(f'invalid unit {u!r}')


def size_mm_to_px(s: gws.Size, ppi: int) -> gws.Size:
    """Convert a size in millimetres to pixels.

    Args:
        s: Size in millimetres.
        ppi: Pixels per inch.

    Returns:
        Size in pixels.
    """
    x, y = s
    return mm_to_px(x, ppi), mm_to_px(y, ppi)


def size_to_px(s: gws.UomSize, ppi: int) -> gws.UomSize:
    """Convert a size with a unit to pixels.

    Args:
        s: Size in ``px`` or ``mm``.
        ppi: Pixels per inch.

    Returns:
        Size in pixels.

    Raises:
        ``ValueError``: If the unit is not ``px`` or ``mm``.
    """
    x, y, u = s
    if u == gws.Uom.px:
        return s
    if u == gws.Uom.mm:
        return mm_to_px(x, ppi), mm_to_px(y, ppi), gws.Uom.px
    raise ValueError(f'invalid unit {u!r}')


def extent_to_px(ext: gws.UomExtent, ppi: int) -> gws.UomExtent:
    """Convert an extent with a unit to pixels.

    Args:
        ext: Extent in ``px`` or ``mm``.
        ppi: Pixels per inch.

    Returns:
        Extent in pixels.

    Raises:
        ``ValueError``: If the unit is not ``px`` or ``mm``.
    """
    a, b, c, d, u = ext
    if u == gws.Uom.px:
        return ext
    if u == gws.Uom.mm:
        return (
            mm_to_px(a, ppi),
            mm_to_px(b, ppi),
            mm_to_px(c, ppi),
            mm_to_px(d, ppi),
            gws.Uom.px,
        )
    raise ValueError(f'invalid unit {u!r}')


##


def px_to_mm(x: _number, ppi: int) -> float:
    """Convert pixels to millimetres.

    Args:
        x: Number of pixels.
        ppi: Pixels per inch.

    Returns:
        Millimetres.
    """
    return x * (MM_PER_IN / ppi)


def to_mm(v: gws.UomValue, ppi: int) -> gws.UomValue:
    """Convert a value with a unit to millimetres.

    Args:
        v: Value in ``mm`` or ``px``.
        ppi: Pixels per inch.

    Returns:
        The value in millimetres.

    Raises:
        ``ValueError``: If the unit is not ``mm`` or ``px``.
    """
    x, u = v
    if u == gws.Uom.mm:
        return v
    if u == gws.Uom.px:
        return px_to_mm(x, ppi), gws.Uom.mm
    raise ValueError(f'invalid unit {u!r}')


def size_px_to_mm(s: gws.Size, ppi: int) -> gws.Size:
    """Convert a size in pixels to millimetres.

    Args:
        s: Size in pixels.
        ppi: Pixels per inch.

    Returns:
        Size in millimetres.
    """
    x, y = s
    return px_to_mm(x, ppi), px_to_mm(y, ppi)


def size_to_mm(s: gws.UomSize, ppi: int) -> gws.UomSize:
    """Convert a size with a unit to millimetres.

    Args:
        s: Size in ``mm`` or ``px``.
        ppi: Pixels per inch.

    Returns:
        Size in millimetres.

    Raises:
        ``ValueError``: If the unit is not ``mm`` or ``px``.
    """
    x, y, u = s
    if u == gws.Uom.mm:
        return s
    if u == gws.Uom.px:
        return px_to_mm(x, ppi), px_to_mm(y, ppi), gws.Uom.mm
    raise ValueError(f'invalid unit {u!r}')


def extent_to_mm(ext: gws.UomExtent, ppi: int) -> gws.UomExtent:
    """Convert an extent with a unit to millimetres.

    Args:
        ext: Extent in ``mm`` or ``px``.
        ppi: Pixels per inch.

    Returns:
        Extent in millimetres.

    Raises:
        ``ValueError``: If the unit is not ``mm`` or ``px``.
    """
    a, b, c, d, u = ext
    if u == gws.Uom.mm:
        return ext
    if u == gws.Uom.px:
        return (
            px_to_mm(a, ppi),
            px_to_mm(b, ppi),
            px_to_mm(c, ppi),
            px_to_mm(d, ppi),
            gws.Uom.mm,
        )
    raise ValueError(f'invalid unit {u!r}')


def to_str(v: gws.UomValue) -> str:
    """Convert a value with a unit to a string.

    Whole numbers are written without a decimal part.

    Args:
        v: Value with a unit.

    Returns:
        A string like ``5mm``.
    """
    x, u = v
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


def parse(val: str | float | tuple | list, default_unit: gws.Uom = None) -> gws.UomValue:
    """Parse a value with a unit.

    Args:
        val: A string like ``'5mm'``, a number, or a pair like ``[5, 'mm']``.
        default_unit: Unit for numbers and for strings without a known unit.

    Returns:
        The value with its unit.

    Raises:
        ``ValueError``: If the format is invalid, or the unit is missing or unknown and there is no default unit.
    """
    if isinstance(val, (list, tuple)):
        if len(val) == 2:
            return parse(f'{val[0]}{val[1]}', default_unit)
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
    """Parse a point with a unit.

    Args:
        val: A comma-separated string like ``'1mm,2mm'``, a list like ``['1mm', '2mm']`` or a list like ``[1, 2, 'mm']``.

    Returns:
        The point with its unit.

    Raises:
        ``ValueError``: If the point is invalid or the units differ.
    """

    v = gws.u.to_list(val)

    if len(v) == 3:
        x, y, u = v
        v = [f'{x}{u}', f'{y}{u}']

    if len(v) == 2:
        x, u1 = parse(v[0])
        y, u2 = parse(v[1])
        if u1 != u2:
            raise ValueError(f'invalid point units: {u1!r} != {u2!r}')
        return x, y, u1

    raise ValueError(f'invalid point: {val!r}')


def parse_extent(val: str | tuple | list) -> gws.UomExtent:
    """Parse an extent with a unit.

    Args:
        val: A comma-separated string like ``'1mm,2mm,3mm,4mm'``, a list of four strings, or a list like ``[1, 2, 3, 4, 'mm']``.

    Returns:
        The extent with its unit.

    Raises:
        ``ValueError``: If the extent is invalid or the units differ.
    """

    v = gws.u.to_list(val)

    if len(v) == 5:
        a, b, c, d, u = v
        v = [
            f'{a}{u}',
            f'{b}{u}',
            f'{c}{u}',
            f'{d}{u}',
        ]

    if len(v) == 4:
        a, b, c, d = v
        n1, u1 = parse(a)
        n2, u2 = parse(b)
        n3, u3 = parse(c)
        n4, u4 = parse(d)
        if u1 != u2 or u1 != u3 or u1 != u4:
            raise ValueError(f'invalid extent units: {u1!r} != {u2!r} != {u3!r} != {u4!r}')
        return n1, n2, n3, n4, u1

    raise ValueError(f'invalid extent: {val!r}')
