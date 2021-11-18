import math
import re

import gws
import gws.gis.crs
import gws.types as t


def from_string(s: str) -> t.Optional[gws.Extent]:
    """Create an extent from a comma-separated string "1000,2000,20000 40000" """

    return from_str_list(s.split(','))


def from_str_list(ls: t.List[str]) -> t.Optional[gws.Extent]:
    """Create an extent from a list of numeric strings"""

    try:
        ns = [float(n) for n in ls]
    except:
        return None

    return _valid(ns)


def from_inverted_str_list(ls: t.List[str]) -> t.Optional[gws.Extent]:
    return from_str_list([ls[1], ls[0], ls[3], ls[2]])


def from_list(ls: t.List[t.Any]) -> t.Optional[gws.Extent]:
    """Create an extent from a list of float values"""

    return _valid(ls)


def from_points(a: gws.Point, b: gws.Point) -> gws.Extent:
    return _valid([a[0], a[1], b[0], b[1]])


def from_center(xy: gws.Point, size: gws.Size) -> gws.Extent:
    return (
        xy[0] - size[0] / 2,
        xy[1] - size[1] / 2,
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
    )


def from_box(box: str) -> t.Optional[gws.Extent]:
    """Create an extent from a Postgis BOX(1000 2000,20000 40000)"""

    if not box:
        return None

    m = re.match(r'^BOX\((.+?)\)$', str(box).upper())
    if not m:
        return None

    try:
        a, b = m.group(1).split(',')
        c, d = a.split(), b.split()
        ls = [
            float(c[0]),
            float(c[1]),
            float(d[0]),
            float(d[1]),
        ]
    except:
        return None

    return _valid(ls)


#

def _valid(ls: t.List[t.Any]) -> t.Optional[gws.Extent]:
    try:
        if len(ls) == 4 and all(math.isfinite(p) for p in ls):
            return _sort(ls)
    except:
        return None


def merge(exts: t.List[gws.Extent]) -> t.Optional[gws.Extent]:
    c = False
    res = (1e20, 1e20, -1e20, -1e20)

    for e in exts:
        e = _sort(e)
        res = (
            min(res[0], e[0]),
            min(res[1], e[1]),
            max(res[2], e[2]),
            max(res[3], e[3])
        )
        c = True

    return res if c else None


def constrain(a: gws.Extent, b: gws.Extent) -> gws.Extent:
    a = _sort(a)
    b = _sort(b)
    return (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )


def center(e: gws.Extent) -> gws.Point:
    return (
        e[0] + (e[2] - e[0]) / 2,
        e[1] + (e[3] - e[1]) / 2,
    )


def size(e: gws.Extent) -> gws.Size:
    return (
        e[2] - e[0],
        e[3] - e[1],
    )


def diagonal(e: gws.Extent) -> float:
    return math.sqrt((e[2] - e[0]) ** 2 + (e[3] - e[1]) ** 2)


def circumsquare(e: gws.Extent) -> gws.Extent:
    """A circumscribed square of the extent."""

    d = diagonal(e)
    return from_center(center(e), (d, d))


def buffer(e: gws.Extent, buf: int) -> gws.Extent:
    e = _sort(e)
    return (
        e[0] - buf,
        e[1] - buf,
        e[2] + buf,
        e[3] + buf,
    )


def intersect(a: gws.Extent, b: gws.Extent) -> bool:
    a = _sort(a)
    b = _sort(b)
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def transform(e: gws.Extent, source: gws.ICrs, target: gws.ICrs) -> gws.Extent:
    return source.transform_extent(e, target)


def transform_to_4326(e: gws.Extent, source: gws.ICrs) -> gws.Extent:
    return source.transform_extent(e, gws.gis.crs.get4326())


def swap_xy(e: gws.Extent) -> gws.Extent:
    return e[1], e[0], e[3], e[2]


def _sort(e):
    # our extents are always [minx, miny, maxx, maxy]
    return (
        min(e[0], e[2]),
        min(e[1], e[3]),
        max(e[0], e[2]),
        max(e[1], e[3]),
    )
