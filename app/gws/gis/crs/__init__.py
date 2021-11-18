import fiona.transform
import math
import os
import osgeo.osr
import pyproj
import re

import gws
import gws.lib.sqlite
import gws.types as t

# osgeo.osr.UseExceptions()

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
# it's imperative that they are singletons

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

    if srid in _cache:
        return _cache[srid]

    rec = _load(srid)

    if not rec:
        _cache[crsid] = _cache[srid] = None
        return None

    crs = Crs(rec)

    with gws.app_lock('crs'):
        if srid in _cache:
            return _cache[srid]
        _cache[crsid] = _cache[srid] = crs

    return _cache[srid]


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

def best_match(crs: gws.ICrs, supported_crs: t.List[gws.ICrs]) -> gws.ICrs:
    """Return a crs from the list that most closely matches the given crs.

    Args:
        crs: target CRS
        supported_crs: CRS list

    Returns:
        A CRS object
    """

    for sup in supported_crs:
        if sup == crs:
            return crs

    bst = _best_match(crs, supported_crs)
    gws.log.debug(f'best_crs: using {bst.srid!r} for {crs.srid!r}')
    return bst


def _best_match(crs, supported_crs):
    # @TODO find a projection with less errors
    # @TODO find a projection with same units

    if crs.is_projected:
        # for a projected crs, find webmercator
        for sup in supported_crs:
            if sup.srid == c3857:
                return sup

        # not found, return the first projected crs
        for sup in supported_crs:
            if sup.is_projected:
                return sup

    if crs.is_geographic:

        # for a geographic crs, try wgs first
        for sup in supported_crs:
            if sup.srid == c4326:
                return sup

        # not found, return the first geographic crs
        for sup in supported_crs:
            if sup.is_geographic:
                return sup

    # should never be here, but...

    for sup in supported_crs:
        return sup


def best_bounds(crs: gws.ICrs, supported_bounds: t.List[gws.Bounds]) -> gws.Bounds:
    """Return the best one from the list of supported bounds.

    Args:
        crs: target CRS
        supported_bounds:

    Returns:
        A Bounds object
    """

    bst = best_match(crs, [b.crs for b in supported_bounds])
    for b in supported_bounds:
        if b.crs == bst:
            return b


def best_axis(
    crs: gws.ICrs,
    protocol: gws.OwsProtocol = None,
    protocol_version: str = None,
    crs_format: gws.CrsFormat = None,
    inverted_crs: t.Optional[t.List[gws.ICrs]] = None
) -> gws.Axis:
    """Return the 'best guess' axis under given circumstances.

    Args:
        crs: target CRS
        protocol: OWS protocol (WMS, WFS...)
        protocol_version: protocol version
        crs_format: the format the target_crs was obtained from
        inverted_crs: user-provided list of CRSes known to have an inverted (YX) axis

    Returns:
        An axis
    """

    # @TODO some logic to guess the axis, based on crs, service protocol and version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis

    if inverted_crs and crs in inverted_crs:
        return gws.AXIS_YX

    return gws.AXIS_XY


##


class Crs(gws.Object, gws.ICrs):
    def __init__(self, rec):
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

        # self.pp = pyproj.Proj(self.epsg)

    def transform_extent(self, ext, target):
        if target == self:
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
        if target == self:
            return geom
        return fiona.transform.transform_geom(self.epsg, target.epsg, geom)

    def to_string(self, fmt):
        return getattr(self, str(fmt).lower())

    def to_geojson(self):
        # https://geojson.org/geojson-spec#named-crs
        return {
            'type': 'name',
            'properties': {
                'name': self.urn,
            }
        }

    def __repr__(self):
        return f'<crs:{self.srid}>'


##

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
