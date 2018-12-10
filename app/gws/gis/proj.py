import re
import os
import threading

import osgeo.osr
import pyproj
import shapely.ops

import sqlite3

import gws

osgeo.osr.UseExceptions()

"""
All functions accept:

int/numeric EPSG Code
4326

EPSG Code 
EPSG:4326

OGC HTTP URL
http://www.opengis.net/gml/srs/epsg.xml#4326

OGC Experimental URN
urn:x-ogc:def:crs:EPSG:4326

OGC URN
urn:ogc:def:crs:EPSG::4326

OGC HTTP URI
http://www.opengis.net/def/crs/EPSG/0/4326

# https://docs.geoserver.org/stable/en/user/services/wfs/webadmin.html#gml


"""


def is_latlong(p) -> bool:
    return _check(p).is_latlong


def units(p) -> str:
    return _check(p).units


def equal(p, q):
    if p == q:
        return True

    p = as_proj(p)
    q = as_proj(q)

    return p and q and p.srid == q.srid


def as_epsg(p):
    return _check(p).epsg


def as_proj4text(p):
    return _check(p).proj4text


def as_urn(p):
    return _check(p).urn


def as_urnx(p):
    return _check(p).urnx


def as_http(p):
    return _check(p).http


def transform_xy(x, y, src, dst):
    src = _check(src)
    dst = _check(dst)

    if src.srid == dst.srid:
        return x, y

    return pyproj.transform(src.py, dst.py, x, y, None)


def transform(geo, src, dst):
    src = _check(src)
    dst = _check(dst)

    if src.srid == dst.srid:
        return geo

    fn = lambda x, y, z=None: pyproj.transform(src.py, dst.py, x, y, z)
    return shapely.ops.transform(fn, geo)


def transform_bbox(bbox, src, dst):
    src = _check(src)
    dst = _check(dst)

    if src.srid == dst.srid:
        return bbox

    a = pyproj.transform(src.py, dst.py, bbox[0], bbox[1])
    b = pyproj.transform(src.py, dst.py, bbox[2], bbox[3])
    return a[0], a[1], b[0], b[1]


##


def find(crs, crs_list):
    for c in crs_list:
        if equal(crs, c):
            return c


##

_formats = {
    'epsg': 'EPSG:%d',
    'http': 'http://www.opengis.net/gml/srs/epsg.xml#%d',
    'urnx': 'urn:x-ogc:def:crs:epsg:%d',
    'urn': 'urn:ogc:def:crs:epsg::%d',
}

_res = {
    'epsg': r'^epsg:(\d+)$',
    'http': r'^http://www.opengis.net/gml/srs/epsg.xml#(\d+)$',
    'urnx': r'^urn:x-ogc:def:crs:epsg:(\d+)$',
    'urn': r'^urn:ogc:def:crs:epsg::(\d+)$',
}

# @TODO

_aliases = {
    'crs:84': 4326,
    'crs84': 4326,
    'urn:ogc:def:crs:OGC:1.3:CRS84': 4326,
    'wgs84': 4326,
    'epsg:900913': 3857,
}


def parse(p):

    if isinstance(p, int):
        return 'srid', p

    if isinstance(p, bytes):
        p = p.decode('ascii')

    if p.isdigit():
        return 'srid', int(p)

    p = p.lower()

    if p in _aliases:
        return 'other', _aliases[p]


    for k, r in _res.items():
        m = re.match(r, p)
        if m:
            return k, int(m.group(1))

    return None, 0


##

_cache = {}
_lock = threading.RLock()


def as_proj(p):
    if isinstance(p, _Proj):
        return p
    if p in _cache:
        return _cache[p]
    gws.log.debug(f'proj: loading {p!r}')
    with _lock:
        _cache[p] = _load_proj(p)
    return _cache[p]


_dbpath = os.path.dirname(__file__) + '/crs.sqlite'


def _load_proj(p):
    fmt, srid = parse(p)
    if not srid:
        gws.log.warn(f'proj: cannot parse {p!r}')
        return

    with sqlite3.connect(_dbpath) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM crs WHERE srid=?', (srid,))
        r = c.fetchone()

    if not r:
        gws.log.warn(f'proj: cannot find {p!r}')
        return

    return _Proj(srid, r[1], r[2], r[3])


class _Proj:
    def __init__(self, srid, proj4text, units, is_latlong):
        self.srid = srid
        self.proj4text = proj4text
        self.units = units or ''
        self.is_latlong = bool(is_latlong)

        self.epsg = _formats['epsg'] % srid
        self.urn = _formats['urn'] % srid
        self.urnx = _formats['urnx'] % srid
        self.http = _formats['http'] % srid

        self.py = pyproj.Proj(init=self.epsg)


def _check(p):
    prj = as_proj(p)
    if not prj:
        raise ValueError('proj: invalid CRS {p!r}')
    return prj
