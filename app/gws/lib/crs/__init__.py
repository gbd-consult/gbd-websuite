"""Coordinate reference systems.

This package provides ``gws.Crs`` objects, which describe coordinate reference systems
and transform extents, points and resolutions between them. CRS objects are created
from EPSG definitions with ``pyproj``, and are cached, so that there is one object per SRID.
Only CRSs with meter or degree units are supported. CRS objects with the same SRID compare equal.

``transform_resolution`` samples nine points of the extent (corners, edge midpoints, centre),
transforms each together with a one-pixel step in x and in y, and returns the smallest
finite positive step length in the target CRS.

A CRS can be referenced by a ``gws.CrsName`` in one of these formats (see ``gws.CrsFormat``):

- numeric SRID: ``4326``
- EPSG code: ``EPSG:4326``
- OGC HTTP URL: ``http://www.opengis.net/gml/srs/epsg.xml#4326``
- OGC experimental URN: ``urn:x-ogc:def:crs:EPSG:4326``
- OGC URN: ``urn:ogc:def:crs:EPSG::4326``
- OGC HTTP URI: ``http://www.opengis.net/def/crs/EPSG/0/4326``

Names are case-insensitive. Some aliases, like ``CRS:84`` or ``EPSG:900913``, are also recognized.

The package provides:

- predefined CRS objects ``WGS84`` and ``WEBMERCATOR`` and related constants,
- ``get``, ``require`` and ``parse`` to look up a CRS by name,
- ``best_match`` to pick a CRS from a list of supported CRSs,
- ``qgis_extent_width`` to compute the width of a geographic extent the way QGIS does.

The ``gws.Crs`` interface itself is defined in ``types.pyinc``.

Example::

    import gws.lib.crs

    crs = gws.lib.crs.require('EPSG:25832')
    crs.to_string(gws.CrsFormat.urn)  # 'urn:ogc:def:crs:EPSG::25832'

    ext = gws.lib.crs.WGS84.transform_extent((5.0, 47.0, 15.0, 55.0), crs)

    fmt, crs = gws.lib.crs.parse('urn:ogc:def:crs:EPSG::4326')
    crs.axis_for_format(fmt)  # gws.Axis.yx
"""

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
    """Coordinate reference system."""

    def __init__(self, **kwargs):
        """Create a CRS object with the given attributes.

        Args:
            **kwargs: Attribute values, see ``gws.Crs``.
        """
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
        return _transform_extent_check(ext, self.srid, crs_to.srid)

    def transform_resolution(self, extent, res, crs_to):
        tr = self.transformer(crs_to)

        x0, y0, x1, y1 = extent
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        points = [
            (x0, y0),
            (xm, y0),
            (x1, y0),
            (x0, ym),
            (xm, ym),
            (x1, ym),
            (x0, y1),
            (xm, y1),
            (x1, y1),
        ]

        ds = []
        for x, y in points:
            ax, ay = tr(x, y)
            bx, by = tr(x + res, y)
            cx, cy = tr(x, y + res)
            ds.append(math.hypot(bx - ax, by - ay))
            ds.append(math.hypot(cx - ax, cy - ay))

        ds = [d for d in ds if math.isfinite(d) and d > 0]
        return min(ds) if ds else 0.0

    def clip_wgs_extent(self, wgs_extent):
        a = wgs_extent
        b = self.wgsMaxExtent
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if x0 >= x1 or y0 >= y1:
            return None
        return x0, y0, x1, y1

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
        fmt = fmt or gws.CrsFormat.epsg
        if fmt == gws.CrsFormat.srid:
            return str(self.srid)
        return getattr(self, str(fmt).lower())

    def to_geojson(self):
        # https://geojson.org/geojson-spec#named-crs
        return {
            'type': 'name',
            'properties': {
                'name': self.urn,
            },
        }


##


def qgis_extent_width(extent: gws.Extent) -> float:
    """Compute the width of a geographic extent in meters, the way QGIS does.

    This is a port of ``QgsScaleCalculator::calculateGeographicDistance`` from QGIS.
    The distance is measured along the middle latitude of the extent.

    Args:
        extent: Extent in degrees.

    Returns:
        The width in meters.
    """
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


##

# enough precision to represent 1cm
COORDINATE_PRECISION_DEG = 7
COORDINATE_PRECISION_M = 2

WGS84: gws.Crs = Object(
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
    wgsMaxExtent=(-180, -90, 180, 90),
    coordinatePrecision=COORDINATE_PRECISION_DEG,
)
"""WGS 84 geographic CRS (EPSG:4326)."""

WGS84.bounds = gws.Bounds(crs=WGS84, extent=WGS84.extent)

WEBMERCATOR: gws.Crs = Object(
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
    wgsMaxExtent=(-180, -85.06, 180, 85.06),
    coordinatePrecision=COORDINATE_PRECISION_M,
)
"""WGS 84 / Pseudo-Mercator CRS (EPSG:3857)."""

WEBMERCATOR.bounds = gws.Bounds(crs=WEBMERCATOR, extent=WEBMERCATOR.extent)

WEBMERCATOR_RADIUS = 6378137
"""Radius of the web mercator sphere (the WGS84 semi-major axis), metres."""

METERS_PER_DEGREE = 2 * math.pi * WEBMERCATOR_RADIUS / 360
"""Metres per degree at the equator; the OGC convention for scale denominators in geographic CRS (WMTS 1.0, 6.1)."""

WEBMERCATOR_SQUARE = (
    -math.pi * WEBMERCATOR_RADIUS,
    -math.pi * WEBMERCATOR_RADIUS,
    +math.pi * WEBMERCATOR_RADIUS,
    +math.pi * WEBMERCATOR_RADIUS,
)
"""Square web mercator extent that covers the whole world width, in meters."""


class Error(gws.Error):
    """CRS error."""

    pass


def get(crs_name: Optional[gws.CrsName]) -> Optional[gws.Crs]:
    """Get the CRS for a given CRS name or SRID.

    Args:
        crs_name: CRS name in any supported format, or an SRID.

    Returns:
        The CRS object, or ``None`` if the name is empty, cannot be parsed or refers to an unsupported CRS.
    """
    if not crs_name:
        return None
    crs, err = _get_crs(crs_name)
    if err:
        gws.log.warning(err)
    return crs


def is_valid(crs_name: Optional[gws.CrsName]) -> bool:
    """Check if a CRS name or SRID refers to a supported CRS, without logging.

    Args:
        crs_name: CRS name in any supported format, or an SRID.

    Returns:
        ``True`` if ``get`` would return a CRS.
    """
    if not crs_name:
        return False
    crs, _ = _get_crs(crs_name)
    return crs is not None


def parse(crs_name: gws.CrsName) -> tuple[gws.CrsFormat, Optional[gws.Crs]]:
    """Parse a CRS name into its format and the CRS itself.

    Args:
        crs_name: CRS name in any supported format, or an SRID.

    Returns:
        A tuple of the name format and the CRS object. If the name cannot be parsed,
        ``(CrsFormat.none, None)``. If the CRS is unknown or unsupported, the CRS is ``None``.
    """
    fmt, srid = _parse(crs_name)
    if not fmt:
        return gws.CrsFormat.none, None
    crs, err = _get_crs(srid)
    if err:
        gws.log.warning(err)
    return fmt, crs


def require(crs_name: gws.CrsName) -> gws.Crs:
    """Get the CRS for a given CRS name or SRID, and fail if there is none.

    Args:
        crs_name: CRS name in any supported format, or an SRID.

    Returns:
        The CRS object.

    Raises:
        ``Error``: If the name cannot be parsed or refers to an unknown or unsupported CRS.
    """
    crs, err = _get_crs(crs_name)
    if not crs:
        raise Error(err)
    return crs


##


def best_match(crs: gws.Crs, supported_crs: list[gws.Crs]) -> gws.Crs:
    """Return a CRS from the list that most closely matches the given CRS.

    If the CRS is in the list, it is returned. Otherwise, for a projected CRS, web mercator
    or the first projected CRS from the list is preferred, and for a geographic CRS,
    WGS84 or the first geographic CRS. Failing that, the first CRS from the list is returned,
    or the given CRS if the list is empty.

    Args:
        crs: Target CRS.
        supported_crs: List of supported CRSs.

    Returns:
        A CRS object.
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


def _get_crs(crs_name) -> tuple[Optional[gws.Crs], str]:
    if crs_name in _obj_cache:
        return _obj_cache[crs_name]

    fmt, srid = _parse(crs_name)
    if not fmt:
        res = None, f'CRS: cannot parse {crs_name!r}'
        _obj_cache[crs_name] = res
        return res

    if srid in _obj_cache:
        _obj_cache[crs_name] = _obj_cache[srid]
        return _obj_cache[srid]

    res = _load_crs(srid)
    _obj_cache[crs_name] = _obj_cache[srid] = res
    return res


def _load_crs(srid) -> tuple[Optional[gws.Crs], str]:
    pp = _pyproj_crs_object(srid)
    if not pp:
        return None, f'CRS: unknown srid {srid!r}'

    au = _axis_and_unit(pp)
    if not au:
        return None, f'CRS: unsupported srid {srid!r}'

    axis, uom = au
    if uom not in (gws.Uom.m, gws.Uom.deg):
        return None, f'CRS: unsupported unit {uom!r} for {srid!r}'

    return _make_crs(srid, pp, axis, uom)


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


def _transform_extent_check(ext, srid_from, srid_to):
    ext_nor = _normalize_extent(ext)

    if srid_from == WGS84.srid:
        ext_wgs = ext_nor
    else:
        tr_to_wgs = _pyproj_transformer(srid_from, WGS84.srid)
        ext_wgs = tr_to_wgs.transform_bounds(ext_nor[0], ext_nor[1], ext_nor[2], ext_nor[3])

    if srid_to == WGS84.srid:
        return _normalize_extent(ext_wgs)

    pp = _pyproj_crs_object(srid_to)
    if not pp:
        raise Error(f'_transform_extent: unknown {srid_to=}')

    tr_from_wgs = _pyproj_transformer(WGS84.srid, srid_to)
    au = pp.area_of_use

    if au:
        ext_au = au.bounds
        if _is_big_extent(ext_wgs) and not _is_big_extent(ext_au):
            ext_to = _transform_extent_sampled(ext_wgs, tr_from_wgs, au)
            if ext_to:
                gws.log.debug(f'transform_extent: {ext=} {srid_from!r}->{srid_to!r}: big extent {ext_to=} ')
                return _normalize_extent(ext_to)

        if ext_wgs[2] < ext_au[0] or ext_wgs[0] > ext_au[2] or ext_wgs[3] < ext_au[1] or ext_wgs[1] > ext_au[3]:
            gws.log.debug(f'transform_extent: {ext=} {srid_from!r}->{srid_to!r}: outside of AoU ')

    ext_to = tr_from_wgs.transform_bounds(ext_wgs[0], ext_wgs[1], ext_wgs[2], ext_wgs[3])

    return _normalize_extent(ext_to)


def _transform_extent_sampled(ext_wgs, tr, au, samples=50):
    x0, y0, x1, y1 = ext_wgs
    ax0, ay0, ax1, ay1 = au.bounds
    mid_lon = (ax0 + ax1) / 2

    xys = []

    # 1) Dense grid sampling within the area of use (accurate core extent)
    for i in range(samples + 1):
        lon = ax0 + (ax1 - ax0) * i / samples
        for j in range(samples + 1):
            lat = ay0 + (ay1 - ay0) * j / samples
            try:
                xys.append(tr.transform(lon, lat, errcheck=True))
            except Exception:
                pass

    # 2) Sample the full latitude range along the central meridian of the AoU
    #    This captures the full Y extent for "the world" in the target projection
    for i in range(samples + 1):
        lat = y0 + (y1 - y0) * i / samples
        try:
            xys.append(tr.transform(mid_lon, lat, errcheck=True))
        except Exception:
            pass

    # 3) Sample several meridians across the full longitude range
    #    to capture the full X spread at various latitudes
    for i in range(samples + 1):
        lon = x0 + (x1 - x0) * i / samples
        for j in range(samples + 1):
            lat = y0 + (y1 - y0) * j / samples
            try:
                xys.append(tr.transform(lon, lat, errcheck=True))
            except Exception:
                pass

    xs = [x for x, y in xys if math.isfinite(x) and math.isfinite(y)]
    ys = [y for x, y in xys if math.isfinite(x) and math.isfinite(y)]

    if not xs:
        return

    return (min(xs), min(ys), max(xs), max(ys))


def _is_big_extent(ext_wgs):
    dx = abs(ext_wgs[2] - ext_wgs[0])
    dy = abs(ext_wgs[3] - ext_wgs[1])
    return dx > 350 or dy > 160


def _transform_extent_direct(ext, srid_from, srid_to):
    tr = _pyproj_transformer(srid_from, srid_to)

    ext_nor = _normalize_extent(ext)

    res = tr.transform_bounds(
        left=ext_nor[0],
        bottom=ext_nor[1],
        right=ext_nor[2],
        top=ext_nor[3],
        errcheck=True,
    )
    return _normalize_extent(res)


def _normalize_extent(ext):
    return (
        min(ext[0], ext[2]),
        min(ext[1], ext[3]),
        max(ext[0], ext[2]),
        max(ext[1], ext[3]),
    )


def _make_crs(srid, pp, axis, uom):
    crs = Object()

    crs.srid = srid

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            crs.proj4text = pp.to_proj4()
        except pyproj.exceptions.CRSError:
            return None, f'CRS: cannot convert {srid!r} to proj4'

    crs.wkt = pp.to_wkt()

    crs.axis = axis
    crs.uom = uom
    crs.coordinatePrecision = COORDINATE_PRECISION_M if uom == gws.Uom.m else COORDINATE_PRECISION_DEG

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
        return None, f'CRS: no bbox for {crs.srid!r}'

    crs.wgsExtent = (
        b['west_longitude'],
        b['south_latitude'],
        b['east_longitude'],
        b['north_latitude'],
    )
    crs.extent = _transform_extent_check(crs.wgsExtent, WGS84.srid, srid)
    crs.bounds = gws.Bounds(extent=crs.extent, crs=crs)

    crs.wgsMaxExtent = _wgs_max_extent(pp, crs.wgsExtent)

    return crs, ''


def _wgs_max_extent(pp, wgs_extent):
    """The datum's area of use, or, for global datums, the own extent widened by its width on each side."""

    geo = pp.geodetic_crs
    au = geo.area_of_use if geo else None
    if au and not _is_big_extent(au.bounds):
        return _normalize_extent(au.bounds)

    x0, y0, x1, y1 = wgs_extent
    w = x1 - x0
    return max(x0 - w, -180), y0, min(x1 + w, 180), y1


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
    WGS84.srid: (WGS84, ''),
    WEBMERCATOR.url: (WEBMERCATOR, ''),
}

_pyproj_cache: dict = {}

_transformer_cache: dict = {}
