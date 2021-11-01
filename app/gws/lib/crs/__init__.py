import fiona.transform
import math
import os
import osgeo.osr
import pyproj
import re

import gws
import gws.lib.sqlite
import gws.types as t

osgeo.osr.UseExceptions()

"""

Projections can be referenced by:


int/numeric SRID
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

# https://epsg.io/4326

c4326 = 4326

# https://epsg.io/3857

c3857 = 3857

c3857_radius = 6378137

c3857_extent = [
    -(math.pi * c3857_radius),
    -(math.pi * c3857_radius),
    +(math.pi * c3857_radius),
    +(math.pi * c3857_radius),
]


##

_cache: dict = {}


# NB since Crs objects are often compared for equality,
# it's imperative that every (format, srid) object is a singleton

def get(crsid: t.Optional[gws.CrsId]) -> t.Optional['Crs']:
    if not crsid:
        return None
    crs = _get(crsid)
    return crs


def _get(crsid):
    if crsid in _cache:
        return _cache[crsid]

    fmt, srid = _parse(crsid)
    if not fmt:
        _cache[crsid] = None
        return None

    key = fmt, srid
    if key in _cache:
        return _cache[key]

    rec = _load(srid)

    if not rec:
        _cache[crsid] = _cache[key] = _cache[srid] = None
        return None

    crs = Crs(rec, fmt)

    with gws.app_lock('crs'):
        if key in _cache:
            return _cache[key]
        _cache[key] = crs

    return _cache[key]


def get3857():
    return get(3857)


def get4326():
    return get(4326)


def require(crsid: gws.CrsId) -> 'Crs':
    crs = get(crsid)
    if not crs:
        raise gws.Error('invalid CRS {crsid!r}')
    return crs


##


class Crs(gws.Object, gws.ICrs):
    def __init__(self, rec, fmt):
        super().__init__()

        self.srid = rec['srid']
        self.proj4text = rec['proj4text']
        self.units = rec['units'] or 'dms'
        self.is_geographic = bool(rec['longlat'])
        self.is_projected = not bool(rec['longlat'])

        self.epsg = _formats[gws.CrsFormat.EPSG] % self.srid
        self.urn = _formats[gws.CrsFormat.URN] % self.srid
        self.urnx = _formats[gws.CrsFormat.URNX] % self.srid
        self.url = _formats[gws.CrsFormat.URL] % self.srid
        self.uri = _formats[gws.CrsFormat.URI] % self.srid

        self.format = fmt

        # self.pp = pyproj.Proj(self.epsg)

    def same_as(self, other):
        return self.srid == other.srid

    def transform_extent(self, ext, target):
        if self.same_as(target):
            return ext

        ax = min(ext[0], ext[2])
        ay = min(ext[1], ext[3])
        bx = max(ext[0], ext[2])
        by = max(ext[1], ext[3])

        sg = {
            'type': 'Polygon',
            'coordinates': [
                [(bx, ay), (bx, by), (ax, by), (ax, ay), (bx, ay)]
            ]
        }

        dg = fiona.transform.transform_geom(self.epsg, target.epsg, sg)
        cc = dg['coordinates'][0]

        return (
            min(cc[0][0], cc[1][0], cc[2][0], cc[3][0], cc[4][0]),
            min(cc[0][1], cc[1][1], cc[2][1], cc[3][1], cc[4][1]),
            max(cc[0][0], cc[1][0], cc[2][0], cc[3][0], cc[4][0]),
            max(cc[0][1], cc[1][1], cc[2][1], cc[3][1], cc[4][1]),
        )

    def transform_geometry(self, geom, target):
        if self.same_as(target):
            return geom
        return fiona.transform.transform_geom(self.epsg, target.epsg, geom)

    def to_string(self, format=None):
        return getattr(self, str(format or self.format).lower())


##

_formats = {
    gws.CrsFormat.EPSG: 'EPSG:%d',
    gws.CrsFormat.URL: 'http://www.opengis.net/gml/srs/epsg.xml#%d',
    gws.CrsFormat.URI: 'http://www.opengis.net/def/crs/epsg/0/%d',
    gws.CrsFormat.URNX: 'urn:x-ogc:def:crs:EPSG:%d',
    gws.CrsFormat.URN: 'urn:ogc:def:crs:EPSG::%d',
}

_parse_formats = {
    gws.CrsFormat.EPSG: r'^epsg:(\d+)$',
    gws.CrsFormat.URL: r'^http://www.opengis.net/gml/srs/epsg.xml#(\d+)$',
    gws.CrsFormat.URI: r'http://www.opengis.net/def/crs/epsg/0/(\d+)$',
    gws.CrsFormat.URNX: r'^urn:x-ogc:def:crs:epsg:(\d+)$',
    gws.CrsFormat.URN: r'^urn:ogc:def:crs:epsg:[0-9.]*:(\d+)$',
}

# @TODO

_aliases = {
    'crs:84': 4326,
    'crs84': 4326,
    'urn:ogc:def:crs:ogc:1.3:crs84': 'urn:ogc:def:crs:epsg::4326',
    'wgs84': 4326,
    'epsg:900913': 3857,
    'epsg:102100': 3857,
    'epsg:102113': 3857,
}


def _parse(crs):
    if isinstance(crs, str):
        crs = crs.lower()
        if crs in _aliases:
            crs = _aliases[crs]

    if isinstance(crs, int):
        return gws.CrsFormat.EPSG, int(crs)

    if isinstance(crs, bytes):
        crs = crs.decode('ascii').lower()

    if crs.isdigit():
        return gws.CrsFormat.EPSG, int(crs)

    for fmt, r in _parse_formats.items():
        m = re.match(r, crs)
        if m:
            return fmt, int(m.group(1))

    return None, 0


##


_dbpath = gws.dirname(__file__) + '/crs.sqlite'


def _load(srid):
    with gws.lib.sqlite.connect(_dbpath) as conn:
        return conn.execute('SELECT * FROM crs WHERE srid=?', (srid,)).fetchone()


##


_invert_axis = {
    'epsg': False,
    'url': False,
    'uri': True,
    'urnx': True,
    'urn': True,
}


def invert_axis(fmt):
    return _invert_axis.get(fmt, False)
