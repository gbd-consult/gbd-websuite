import math
import re

import fiona.transform

import gws.gis.proj
import gws.gis.shape

import gws.types as t


#:export
class Bounds(t.Data):
    crs: t.Crs
    extent: t.Extent


def from_box(box: str) -> t.Optional[t.Extent]:
    """Create an extent from a Postgis BOX(1000 2000,20000 40000)"""

    if not box:
        return None

    m = re.match(r'^BOX\((.+?)\)$', str(box).upper())
    if not m:
        return None

    try:
        a, b = m.group(1).split(',')
        ext = (
            float(a.split()[0]),
            float(a.split()[1]),
            float(b.split()[0]),
            float(b.split()[1]),
        )
    except:
        return None

    if len(ext) != 4:
        return None

    return valid(ext)


def valid(e: t.Extent) -> t.Optional[t.Extent]:
    if not e:
        return

    if not all(math.isfinite(p) for p in e):
        return

    return _sort(e)


def list_valid(exts: t.List[t.Extent]) -> t.List[t.Extent]:
    ls = []

    for e in exts:
        e = valid(e)
        if e:
            ls.append(e)

    return ls


def merge(exts: t.List[t.Extent]) -> t.Extent:
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


def constrain(a: t.Extent, b: t.Extent) -> t.Extent:
    a = _sort(a)
    b = _sort(b)
    return (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )


def buffer(e: t.Extent, buf: int) -> t.Extent:
    e = _sort(e)
    return (
        e[0] - buf,
        e[1] - buf,
        e[2] + buf,
        e[3] + buf,
    )


def intersect(a: t.Extent, b: t.Extent) -> bool:
    a = _sort(a)
    b = _sort(b)
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def transformed(e: t.Extent, src: t.Crs, dst: t.Crs) -> t.Extent:
    if gws.gis.proj.equal(src, dst):
        return e

    src = gws.gis.proj.as_proj(src)
    dst = gws.gis.proj.as_proj(dst)

    ax, ay, bx, by = _sort(e)

    sg = {
        'type': 'Polygon',
        'coordinates': [
            [(bx, ay), (bx, by), (ax, by), (ax, ay), (bx, ay)]
        ]
    }

    dg = fiona.transform.transform_geom(src.epsg, dst.epsg, sg)
    cc = dg['coordinates'][0]

    return (
        min(cc[0][0], cc[1][0], cc[2][0], cc[3][0]),
        min(cc[0][1], cc[1][1], cc[2][1], cc[3][1]),
        max(cc[0][0], cc[1][0], cc[2][0], cc[3][0]),
        max(cc[0][1], cc[1][1], cc[2][1], cc[3][1]),
    )


def _sort(e):
    # our extents are always [minx, miny, maxx, maxy]
    return (
        min(e[0], e[2]),
        min(e[1], e[3]),
        max(e[0], e[2]),
        max(e[1], e[3]),
    )
