from typing import Optional

import math
import re
import warnings

import pyproj.crs
import pyproj.exceptions
import pyproj.transformer

import gws


##


class Object(gws.Crs):
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    # crs objects with the same srid must be equal
    # (despite caching, they can be different due to pickling)

    def __hash__(self):
        return self.srid

    def __eq__(self, other):
        return isinstance(other, Object) and other.srid == self.srid

    def __repr__(self):
        return f'<crs:{self.srid}>'

    def axis_for_format(self, fmt):
        if not self.isYX:
            return self.axis
        return _AXIS_FOR_FORMAT.get(fmt, self.axis)

    def transform_extent(self, ext, crs_to):
        if crs_to == self:
            return ext
        return _transform_extent(ext, self.srid, crs_to.srid)

    def transformer(self, crs_to):
        tr = _pyproj_transformer(self.srid, crs_to.srid)
        return tr.transform

    def extent_size_in_meters(self, extent):
        x0, y0, x1, y1 = extent

        if self.isProjected:
            if self.uom != gws.Uom.m:
                # @TODO support non-meter crs
                raise Error(f'unsupported unit: {self.uom}')
            return abs(x1 - x0), abs(y1 - y0)

        geod = pyproj.Geod(ellps='WGS84')

        mid_lat = (y0 + y1) / 2
        _, _, w = geod.inv(x0, mid_lat, x1, mid_lat)
        mid_lon = (x0 + x1) / 2
        _, _, h = geod.inv(mid_lon, y0, mid_lon, y1)

        return w, h

    def point_offset_in_meters(self, xy, dist, az):
        x, y = xy

        if self.isProjected:
            if self.uom != gws.Uom.m:
                # @TODO support non-meter crs
                raise Error(f'unsupported unit: {self.uom}')

            if az == 0:
                return x, y + dist
            if az == 90:
                return x + dist, y
            if az == 180:
                return x, y - dist
            if az == 270:
                return x - dist, y

            az_rad = math.radians(90 - az)
            return (
                x + dist * math.cos(az_rad),
                y + dist * math.sin(az_rad),
            )

        geod = pyproj.Geod(ellps='WGS84')
        x, y, _ = geod.fwd(x, y, dist=dist, az=az)
        return x, y

    def to_string(self, fmt=None):
        return getattr(self, str(fmt or gws.CrsFormat.epsg).lower())

    def to_geojson(self):
        # https://geojson.org/geojson-spec#named-crs
        return {
            'type': 'name',
            'properties': {
                'name': self.urn,
            },
        }


#


def qgis_extent_width(extent: gws.Extent) -> float:
    # straight port from QGIS/src/core/qgsscalecalculator.cpp QgsScaleCalculator::calculateGeographicDistance
    x0, y0, x1, y1 = extent

    lat = (y0 + y1) * 0.5
    RADS = (4.0 * math.atan(1.0)) / 180.0
    a = math.pow(math.cos(lat * RADS), 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    RA = 6378000
    E = 0.0810820288
    radius = RA * (1.0 - E * E) / math.pow(1.0 - E * E * math.sin(lat * RADS) * math.sin(lat * RADS), 1.5)
    return (x1 - x0) / 180.0 * radius * c


#

WGS84 = Object(
    srid=4326,
    proj4text='+proj=longlat +datum=WGS84 +no_defs +type=crs',
    wkt='GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
    axis=gws.Axis.yx,
    uom=gws.Uom.deg,
    isGeographic=True,
    isProjected=False,
    isYX=True,
    epsg='EPSG:4326',
    urn='urn:ogc:def:crs:EPSG::4326',
    urnx='urn:x-ogc:def:crs:EPSG:4326',
    url='http://www.opengis.net/gml/srs/epsg.xml#4326',
    uri='http://www.opengis.net/def/crs/epsg/0/4326',
    name='WGS 84',
    base=0,
    datum='World Geodetic System 1984 ensemble',
    wgsExtent=(-180, -90, 180, 90),
    extent=(-180, -90, 180, 90),
)

WGS84.bounds = gws.Bounds(crs=WGS84, extent=WGS84.extent)

WEBMERCATOR = Object(
    srid=3857,
    proj4text='+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs',
    wkt='PROJCRS["WGS 84 / Pseudo-Mercator",BASEGEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["Popular Visualisation Pseudo-Mercator",METHOD["Popular Visualisation Pseudo Mercator",ID["EPSG",1024]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["northing (Y)",north,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Web mapping and visualisation."],AREA["World between 85.06°S and 85.06°N."],BBOX[-85.06,-180,85.06,180]],ID["EPSG",3857]]',
    axis=gws.Axis.xy,
    uom=gws.Uom.m,
    isGeographic=False,
    isProjected=True,
    isYX=False,
    epsg='EPSG:3857',
    urn='urn:ogc:def:crs:EPSG::3857',
    urnx='urn:x-ogc:def:crs:EPSG:3857',
    url='http://www.opengis.net/gml/srs/epsg.xml#3857',
    uri='http://www.opengis.net/def/crs/epsg/0/3857',
    name='WGS 84 / Pseudo-Mercator',
    base=4326,
    datum='World Geodetic System 1984 ensemble',
    wgsExtent=(-180, -85.06, 180, 85.06),
    extent=(
        -20037508.342789244,
        -20048966.104014598,
        20037508.342789244,
        20048966.104014598,
    ),
)

WEBMERCATOR.bounds = gws.Bounds(crs=WEBMERCATOR, extent=WEBMERCATOR.extent)

WEBMERCATOR_RADIUS = 6378137
WEBMERCATOR_SQUARE = (
    -math.pi * WEBMERCATOR_RADIUS,
    -math.pi * WEBMERCATOR_RADIUS,
    +math.pi * WEBMERCATOR_RADIUS,
    +math.pi * WEBMERCATOR_RADIUS,
)


class Error(gws.Error):
    pass


def get(crs_name: Optional[gws.CrsName]) -> Optional[gws.Crs]:
    """Returns the CRS for a given CRS-code or SRID."""
    if not crs_name:
        return None
    return _get_crs(crs_name)


def parse(crs_name: gws.CrsName) -> tuple[gws.CrsFormat, Optional[gws.Crs]]:
    """Parses a CRS to a tuple of CRS-format and the CRS itself."""
    fmt, srid = _parse(crs_name)
    if not fmt:
        return gws.CrsFormat.none, None
    return fmt, _get_crs(srid)


def require(crs_name: gws.CrsName) -> gws.Crs:
    """Raises an error if no correct CRS is given."""
    crs = _get_crs(crs_name)
    if not crs:
        raise Error(f'invalid CRS {crs_name!r}')
    return crs


##


def best_match(crs: gws.Crs, supported_crs: list[gws.Crs]) -> gws.Crs:
    """Return a crs from the list that most closely matches the given crs.

    Args:
        crs: target CRS
        supported_crs: CRS list

    Returns:
        A CRS object
    """

    if crs in supported_crs:
        return crs

    bst = _best_match(crs, supported_crs)
    if not bst:
        bst = supported_crs[0] if supported_crs else crs
    gws.log.debug(f'CRS: best_crs: using {bst.srid!r} for {crs.srid!r}')
    return bst


def _best_match(crs, supported_crs):
    # @TODO find a projection with less errors
    # @TODO find a projection with same units

    if crs.isProjected:
        # for a projected crs, find webmercator
        for sup in supported_crs:
            if sup.srid == WEBMERCATOR.srid:
                return sup

        # not found, return the first projected crs
        for sup in supported_crs:
            if sup.isProjected:
                return sup

    if crs.isGeographic:
        # for a geographic crs, try wgs first
        for sup in supported_crs:
            if sup.srid == WGS84.srid:
                return sup

        # not found, return the first geographic crs
        for sup in supported_crs:
            if sup.isGeographic:
                return sup


##


def _get_crs(crs_name):
    if crs_name in _obj_cache:
        return _obj_cache[crs_name]

    fmt, srid = _parse(crs_name)
    if not fmt:
        _obj_cache[crs_name] = None
        return None

    if srid in _obj_cache:
        _obj_cache[crs_name] = _obj_cache[srid]
        return _obj_cache[srid]

    obj = _get_new_crs(srid)
    _obj_cache[crs_name] = _obj_cache[srid] = obj
    return obj


def _get_new_crs(srid):
    pp = _pyproj_crs_object(srid)
    if not pp:
        gws.log.error(f'CRS: unknown srid {srid!r}')
        return None

    au = _axis_and_unit(pp)
    if not au:
        gws.log.error(f'CRS: unsupported srid {srid!r}')
        return None

    return _make_crs(srid, pp, au)


def _pyproj_crs_object(srid) -> Optional[pyproj.CRS]:
    if srid in _pyproj_cache:
        return _pyproj_cache[srid]

    try:
        pp = pyproj.CRS.from_epsg(srid)
    except pyproj.exceptions.CRSError:
        return None

    _pyproj_cache[srid] = pp
    return _pyproj_cache[srid]


def _pyproj_transformer(srid_from, srid_to) -> pyproj.transformer.Transformer:
    key = srid_from, srid_to

    if key in _transformer_cache:
        return _transformer_cache[key]

    pa = _pyproj_crs_object(srid_from)
    pb = _pyproj_crs_object(srid_to)

    _transformer_cache[key] = pyproj.transformer.Transformer.from_crs(pa, pb, always_xy=True)
    return _transformer_cache[key]


def _transform_extent(ext, srid_from, srid_to):
    tr = _pyproj_transformer(srid_from, srid_to)

    e = tr.transform_bounds(
        left=min(ext[0], ext[2]),
        bottom=min(ext[1], ext[3]),
        right=max(ext[0], ext[2]),
        top=max(ext[1], ext[3]),
    )

    return (
        min(e[0], e[2]),
        min(e[1], e[3]),
        max(e[0], e[2]),
        max(e[1], e[3]),
    )


def _make_crs(srid, pp, au):
    crs = Object()

    crs.srid = srid

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            crs.proj4text = pp.to_proj4()
        except pyproj.exceptions.CRSError:
            gws.log.error(f'CRS: cannot convert {srid!r} to proj4')
            return None

    crs.wkt = pp.to_wkt()

    crs.axis = au[0]
    crs.uom = au[1]

    crs.isGeographic = pp.is_geographic
    crs.isProjected = pp.is_projected
    crs.isYX = crs.axis == gws.Axis.yx

    crs.epsg = _unparse(crs.srid, gws.CrsFormat.epsg)
    crs.urn = _unparse(crs.srid, gws.CrsFormat.urn)
    crs.urnx = _unparse(crs.srid, gws.CrsFormat.urnx)
    crs.url = _unparse(crs.srid, gws.CrsFormat.url)
    crs.uri = _unparse(crs.srid, gws.CrsFormat.uri)

    # see https://proj.org/schemas/v0.5/projjson.schema.json
    d = pp.to_json_dict()

    crs.name = d.get('name') or str(crs.srid)

    def _datum(x):
        if 'datum_ensemble' in x:
            return x['datum_ensemble']['name']
        if 'datum' in x:
            return x['datum']['name']
        return ''

    def _bbox(d):
        b = d.get('bbox')
        if b:
            # pyproj 3.6
            return b
        if d.get('usages'):
            # pyproj 3.7
            for u in d['usages']:
                b = u.get('bbox')
                if b:
                    return b

    b = d.get('base_crs')
    if b:
        crs.base = b['id']['code']
        crs.datum = _datum(b)
    else:
        crs.base = 0
        crs.datum = _datum(d)

    b = _bbox(d)
    if not b:
        gws.log.error(f'CRS: no bbox for {crs.srid!r}')
        return

    crs.wgsExtent = (
        b['west_longitude'],
        b['south_latitude'],
        b['east_longitude'],
        b['north_latitude'],
    )
    crs.extent = _transform_extent(crs.wgsExtent, WGS84.srid, srid)
    crs.bounds = gws.Bounds(extent=crs.extent, crs=crs)

    return crs


_AXES_AND_UNITS = {
    'Easting/metre,Northing/metre': (gws.Axis.xy, gws.Uom.m),
    'Northing/metre,Easting/metre': (gws.Axis.yx, gws.Uom.m),
    'Geodetic latitude/degree,Geodetic longitude/degree': (gws.Axis.yx, gws.Uom.deg),
    'Geodetic longitude/degree,Geodetic latitude/degree': (gws.Axis.xy, gws.Uom.deg),
    'Easting/US survey foot,Northing/US survey foot': (gws.Axis.xy, gws.Uom.us_ft),
    'Easting/foot,Northing/foot': (gws.Axis.xy, gws.Uom.ft),
}


def _axis_and_unit(pp):
    ax = []
    for a in pp.axis_info:
        ax.append(a.name + '/' + a.unit_name)
    return _AXES_AND_UNITS.get(','.join(ax))


##

"""
Projections can be referenced by:

    - int/numeric SRID: 4326
    - EPSG Code: EPSG:4326
    - OGC HTTP URL: http://www.opengis.net/gml/srs/epsg.xml#4326
    - OGC Experimental URN: urn:x-ogc:def:crs:EPSG:4326
    - OGC URN: urn:ogc:def:crs:EPSG::4326
    - OGC HTTP URI: http://www.opengis.net/def/crs/EPSG/0/4326

# https://docs.geoserver.org/stable/en/user/services/wfs/webadmin.html#gml
"""

_WRITE_FORMATS = {
    gws.CrsFormat.srid: '{:d}',
    gws.CrsFormat.epsg: 'EPSG:{:d}',
    gws.CrsFormat.url: 'http://www.opengis.net/gml/srs/epsg.xml#{:d}',
    gws.CrsFormat.uri: 'http://www.opengis.net/def/crs/epsg/0/{:d}',
    gws.CrsFormat.urnx: 'urn:x-ogc:def:crs:EPSG:{:d}',
    gws.CrsFormat.urn: 'urn:ogc:def:crs:EPSG::{:d}',
}

_PARSE_FORMATS = {
    gws.CrsFormat.srid: r'^(\d+)$',
    gws.CrsFormat.epsg: r'^epsg:(\d+)$',
    gws.CrsFormat.url: r'^http://www.opengis.net/gml/srs/epsg.xml#(\d+)$',
    gws.CrsFormat.uri: r'http://www.opengis.net/def/crs/epsg/0/(\d+)$',
    gws.CrsFormat.urnx: r'^urn:x-ogc:def:crs:epsg:(\d+)$',
    gws.CrsFormat.urn: r'^urn:ogc:def:crs:epsg:[0-9.]*:(\d+)$',
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

# https://docs.geoserver.org/latest/en/user/services/wfs/axis_order.html
# EPSG:4326                                    longitude/latitude
# http://www.opengis.net/gml/srs/epsg.xml#xxxx longitude/latitude
# urn:x-ogc:def:crs:EPSG:xxxx                  latitude/longitude
# urn:ogc:def:crs:EPSG::4326                   latitude/longitude

_AXIS_FOR_FORMAT = {
    gws.CrsFormat.srid: gws.Axis.xy,
    gws.CrsFormat.epsg: gws.Axis.xy,
    gws.CrsFormat.url: gws.Axis.xy,
    gws.CrsFormat.uri: gws.Axis.xy,
    gws.CrsFormat.urnx: gws.Axis.yx,
    gws.CrsFormat.urn: gws.Axis.yx,
}


def _parse(crs_name):
    if isinstance(crs_name, int):
        return gws.CrsFormat.epsg, crs_name

    if isinstance(crs_name, bytes):
        crs_name = crs_name.decode('ascii').lower()

    if isinstance(crs_name, str):
        crs_name = crs_name.lower()

        if crs_name in {'crs84', 'crs:84'}:
            return gws.CrsFormat.crs, 4326

        if crs_name in _aliases:
            crs_name = _aliases[crs_name]
            if isinstance(crs_name, int):
                return gws.CrsFormat.epsg, int(crs_name)

        for fmt, r in _PARSE_FORMATS.items():
            m = re.match(r, crs_name)
            if m:
                return fmt, int(m.group(1))

    return None, 0


def _unparse(srid, fmt):
    return _WRITE_FORMATS[fmt].format(srid)


##


_obj_cache: dict = {
    WGS84.srid: WGS84,
    WEBMERCATOR.url: WEBMERCATOR,
}

_pyproj_cache: dict = {}

_transformer_cache: dict = {}
