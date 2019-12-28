import math
import re


def from_box(box):
    """Create an extent from a Postgis BOX(1000 2000,20000 40000)"""

    if not box:
        return None

    m = re.match(r'^BOX\((.+?)\)$', str(box).upper())
    if not m:
        return None

    try:
        a, b = m.group(1).split(',')
        ext = [
            float(a.split()[0]),
            float(a.split()[1]),
            float(b.split()[0]),
            float(b.split()[1]),
        ]
    except:
        return None

    if len(ext) != 4:
        return None

    return valid(ext)


def valid(e):
    if not e:
        return

    if not all(math.isfinite(p) for p in e):
        return

    minx = min(e[0], e[2])
    maxx = max(e[0], e[2])
    miny = min(e[1], e[3])
    maxy = max(e[1], e[3])

    return [minx, miny, maxx, maxy]


def list_valid(exts):
    ls = []

    for e in exts:
        e = valid(e)
        if e:
            ls.append(e)

    return ls


def merge(exts):
    c = False
    res = [1e20, 1e20, -1e20, -1e20]

    for e in exts:
        res = [
            min(res[0], e[0]),
            min(res[1], e[1]),
            max(res[2], e[2]),
            max(res[3], e[3])
        ]
        c = True

    return res if c else None


def constrain(a, b):
    return [
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    ]


def buffer(e, buf):
    return [
        e[0] - buf,
        e[1] - buf,
        e[2] + buf,
        e[3] + buf,
    ]


def intersect(a, b):
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]
