import math
import re

import fiona.transform

import gws
import gws.types as t
import gws.lib.proj
import gws.lib.shape


def from_string(s: str) -> t.Optional[gws.Extent]:
    """Create an extent from a comma-separated string "1000,2000,20000 40000" """

    try:
        ls = [float(n) for n in s.split(',')]
    except:
        return None

    return _valid(ls)


def from_list(ls: t.List[t.Any]) -> t.Optional[gws.Extent]:
    """Create an extent from a list of float values"""

    return _valid(ls)


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


def transform(e: gws.Extent, src: str, dst: str) -> gws.Extent:
    if gws.lib.proj.equal(src, dst):
        return e

    src_proj = gws.lib.proj.as_proj(src)
    dst_proj = gws.lib.proj.as_proj(dst)

    ax, ay, bx, by = _sort(e)

    sg = {
        'type': 'Polygon',
        'coordinates': [
            [(bx, ay), (bx, by), (ax, by), (ax, ay), (bx, ay)]
        ]
    }

    dg = fiona.transform.transform_geom(src_proj.epsg, dst_proj.epsg, sg)
    cc = dg['coordinates'][0]

    return (
        min(cc[0][0], cc[1][0], cc[2][0], cc[3][0], cc[4][0]),
        min(cc[0][1], cc[1][1], cc[2][1], cc[3][1], cc[4][1]),
        max(cc[0][0], cc[1][0], cc[2][0], cc[3][0], cc[4][0]),
        max(cc[0][1], cc[1][1], cc[2][1], cc[3][1], cc[4][1]),
    )


def transform_to_4326(e: gws.Extent, crs: str) -> gws.Extent:
    e = transform(e, crs, gws.EPSG_4326)
    return (
        round(e[0], 5),
        round(e[1], 5),
        round(e[2], 5),
        round(e[3], 5),
    )


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