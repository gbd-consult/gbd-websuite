"""Shapes.

A shape (``gws.Shape``) is a geo-referenced geometry: a Shapely geometry
together with a ``gws.Crs``. Shapes are used for feature geometries, search
geometries and map extents throughout the application.

The package is a single module. It provides:

- constructors that create a ``Shape`` from WKT and EWKT, WKB and EWKB (binary or
  hex), SQLAlchemy/GeoAlchemy WKB elements, GeoJSON geometries, shape props or
  dicts, extents, ``gws.Bounds`` and x/y coordinates,
- the ``Shape`` class, which implements the ``gws.Shape`` interface: conversion to
  WKB, WKT, GeoJSON and props, spatial predicates, union and intersection, type
  conversions, buffering with a tolerance and transformation to other CRS,
- the ``Props`` class for shapes sent to and from the client.

Constructors raise ``Error`` if the input cannot be parsed or has no CRS.
EWKT and EWKB inputs carry their own SRID; for plain WKT and WKB a default CRS
must be given. GeoJSON inputs and extents are expected in the axis order of the
CRS, unless ``always_xy`` is set. Circles (``{"type": "Circle", "center": ...,
"radius": ...}``), as sent by the client, are converted to polygons.

Binary predicates and set operations transform the other shape to the CRS of
this shape first. ``to_geojson`` transforms to WGS84 unless asked to keep the CRS.

Example::

    import gws.lib.shape
    import gws.lib.crs

    shape = gws.lib.shape.from_wkt('POINT(10 20)', gws.lib.crs.WGS84)
    area = shape.tolerance_polygon(5).transformed_to(gws.lib.crs.WEBMERCATOR)
    ewkt = area.to_ewkt()

    other = gws.lib.shape.from_wkt('SRID=4326;POLYGON((0 0,30 0,30 30,0 30,0 0))')
    print(other.contains(shape))
"""

# @TODO support for SQL/MM extensions

import struct
import re
import shapely.errors
import shapely.geometry
import shapely.ops
import shapely.wkb
import shapely.wkt

import gws
import gws.lib.crs
import gws.lib.sa as sa

_TOLERANCE_QUAD_SEGS = 6
_MIN_TOLERANCE_RADIUS = 0.01


class Error(gws.Error):
    """Invalid geometry or CRS."""

    pass


def from_wkt(wkt: str, default_crs: gws.Crs = None) -> gws.Shape:
    """Create a shape from a WKT or EWKT string.

    Args:
        wkt: A WKT or EWKT string.
        default_crs: CRS to use if the string has no SRID.

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the string is invalid or there is no CRS.
    """

    if wkt.startswith('SRID='):
        # EWKT
        c = wkt.index(';')
        srid = wkt[len('SRID=') : c]
        crs = gws.lib.crs.require(int(srid))
        wkt = wkt[c + 1 :]
    elif default_crs:
        crs = default_crs
    else:
        raise Error('missing or invalid crs for WKT')

    try:
        geom = shapely.wkt.loads(wkt)
    except shapely.errors.ShapelyError as exc:
        raise Error('invalid WKT') from exc
    return Shape(geom, crs)


def from_wkb(wkb: bytes, default_crs: gws.Crs = None) -> gws.Shape:
    """Create a shape from a WKB or EWKB byte string.

    Args:
        wkb: A WKB or EWKB byte string.
        default_crs: CRS to use if the data has no SRID.

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the data is invalid or there is no CRS.
    """

    return _from_wkb(wkb, default_crs)


def from_wkb_hex(wkb: str, default_crs: gws.Crs = None) -> gws.Shape:
    """Create a shape from a hex-encoded WKB or EWKB string.

    Args:
        wkb: A hex-encoded WKB or EWKB string.
        default_crs: CRS to use if the data has no SRID.

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the data is invalid or there is no CRS.
    """

    try:
        b = bytes.fromhex(wkb)
    except ValueError as exc:
        raise Error('invalid WKB hex') from exc
    return _from_wkb(b, default_crs)


def _from_wkb(wkb: bytes, default_crs):
    """Create a shape from WKB or EWKB bytes, reading the SRID from the EWKB header."""
    # http://libgeos.org/specifications/wkb/#extended-wkb

    try:
        byte_order = wkb[0]
        header = struct.unpack('<cLL' if byte_order == 1 else '>cLL', wkb[:9])
    except (IndexError, struct.error) as exc:
        raise Error('invalid WKB') from exc

    if header[1] & 0x20000000:
        crs = gws.lib.crs.require(header[2])
    elif default_crs:
        crs = default_crs
    else:
        raise Error('missing or invalid crs for WKB')

    try:
        geom = shapely.wkb.loads(wkb)
    except shapely.errors.ShapelyError as exc:
        raise Error('invalid WKB') from exc
    return Shape(geom, crs)


def from_wkb_element(element: sa.geo.WKBElement, default_crs: gws.Crs = None):
    """Create a shape from a GeoAlchemy WKB element.

    The CRS is taken from the EWKB data, then from the SRID of the element, then
    from ``default_crs``.

    Args:
        element: A WKB element, with binary or hex-encoded data.
        default_crs: CRS to use if neither the data nor the element has a valid SRID.

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the data is invalid or there is no CRS.
    """
    data = element.data
    if isinstance(data, str):
        wkb = bytes.fromhex(data)
    else:
        wkb = bytes(data)
    crs = gws.lib.crs.get(element.srid)
    return _from_wkb(wkb, crs or default_crs)


def from_geojson(geojson: dict, crs: gws.Crs, always_xy=False) -> gws.Shape:
    """Create a shape from a GeoJSON geometry dict.

    Parses a dict as a GeoJSON geometry object (https://www.rfc-editor.org/rfc/rfc7946#section-3.1).
    A ``Circle`` geometry with ``center`` and ``radius`` is converted to a polygon.
    The coordinates are assumed to be in the axis order of the CRS, unless ``always_xy`` is ``True``.

    Args:
        geojson: A GeoJSON geometry dict.
        crs: A Crs object.
        always_xy: If ``True``, coordinates are assumed to be in the XY (lon/lat) order.

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the geometry is invalid.
    """

    geom = _shapely_shape(geojson)
    if crs.isYX and not always_xy:
        geom = _swap_xy(geom)
    return Shape(geom, crs)


def from_props(props: gws.Props) -> gws.Shape:
    """Create a shape from a properties object.

    Args:
        props: A properties object with ``crs`` and ``geometry`` (a GeoJSON geometry dict).

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the CRS or the geometry is invalid.
    """

    crs = gws.lib.crs.get(props.get('crs'))
    if not crs:
        raise Error('missing or invalid crs')
    geom = _shapely_shape(props.get('geometry'))
    return Shape(geom, crs)


def from_dict(d: dict) -> gws.Shape:
    """Create a shape from a dictionary.

    Args:
        d: A dictionary with the keys ``crs`` and ``geometry`` (a GeoJSON geometry dict).

    Returns:
        A Shape object.

    Raises:
        ``Error``: If the CRS or the geometry is invalid.
    """

    crs = gws.lib.crs.get(d.get('crs'))
    if not crs:
        raise Error('missing or invalid crs')
    geom = _shapely_shape(d.get('geometry'))
    return Shape(geom, crs)


def from_extent(extent: gws.Extent, crs: gws.Crs, always_xy=False) -> gws.Shape:
    """Create a polygon shape from an extent.

    Args:
        extent: An extent.
        crs: A Crs object.
        always_xy: If ``True``, the extent is assumed to be in the XY (lon/lat) order,
            otherwise in the axis order of the CRS.

    Returns:
        A Shape object.
    """

    geom = shapely.geometry.box(*extent)
    if crs.isYX and not always_xy:
        geom = _swap_xy(geom)
    return Shape(geom, crs)


def from_bounds(bounds: gws.Bounds) -> gws.Shape:
    """Create a polygon shape from a Bounds object.

    Args:
        bounds: A Bounds object.

    Returns:
        A Shape object.
    """

    return Shape(shapely.geometry.box(*bounds.extent), bounds.crs)


def from_xy(x: float, y: float, crs: gws.Crs) -> gws.Shape:
    """Create a point shape from coordinates.

    Args:
        x: X coordinate (lon/easting).
        y: Y coordinate (lat/northing).
        crs: A Crs object.

    Returns:
        A Shape object.
    """

    return Shape(shapely.geometry.Point(x, y), crs)


def _swap_xy(geom):
    """Return a copy of a Shapely geometry with x and y swapped."""
    def f(x: float, y: float, z: float = None) -> tuple[float, float]:
        return y, x

    return shapely.ops.transform(f, geom)


_CIRCLE_RESOLUTION = 64


def _shapely_shape(d):
    """Create a Shapely geometry from a GeoJSON dict, raising ``Error`` if it is invalid."""
    try:
        return _shapely_shape2(d)
    except (shapely.errors.ShapelyError, AttributeError, TypeError, ValueError) as exc:
        raise Error('invalid geometry') from exc


def _shapely_shape2(d):
    """Create a Shapely geometry from a GeoJSON dict, converting circles to polygons."""
    if d.get('type').upper() == 'CIRCLE':
        geom = shapely.geometry.Point(d.get('center'))
        return geom.buffer(
            d.get('radius'),
            resolution=_CIRCLE_RESOLUTION,
            cap_style=shapely.geometry.CAP_STYLE.round,
            join_style=shapely.geometry.JOIN_STYLE.round,
        )

    return shapely.geometry.shape(d)


##


class Props(gws.Props):
    """Shape properties object."""

    crs: str
    geometry: dict


##


class Shape(gws.Shape):
    """Shape implemented with a Shapely geometry."""

    geom: shapely.geometry.base.BaseGeometry
    """Shapely geometry."""

    def __init__(self, geom, crs: gws.Crs):
        """Create a shape.

        Args:
            geom: Shapely geometry.
            crs: CRS of the geometry.
        """
        super().__init__()
        self.geom = geom
        self.crs = crs
        self.type = self.geom.geom_type.lower()
        self.x = getattr(self.geom, 'x', None)
        self.y = getattr(self.geom, 'y', None)

    def __str__(self):
        return '{Geometry:' + self.geom.geom_type.upper() + '}'

    def area(self):
        return getattr(self.geom, 'area', 0)

    def bounds(self):
        return gws.Bounds(crs=self.crs, extent=self.geom.bounds)

    def centroid(self):
        return Shape(self.geom.centroid, self.crs)

    def center(self):
        c = self.geom.centroid
        return c.x, c.y

    def to_wkb(self):
        return shapely.wkb.dumps(self.geom)

    def to_wkb_hex(self):
        return shapely.wkb.dumps(self.geom, hex=True)

    def to_ewkb(self):
        return shapely.wkb.dumps(self.geom, srid=self.crs.srid)

    def to_ewkb_hex(self):
        return shapely.wkb.dumps(self.geom, srid=self.crs.srid, hex=True)

    def to_wkt(self, trim=False, rounding_precision=-1, output_dimension=3):
        s = shapely.wkt.dumps(self.geom, trim=trim, rounding_precision=rounding_precision, output_dimension=output_dimension)
        s = re.sub(r'\s*([,()])\s*', r'\1', s)
        s = re.sub(r'\s+', ' ', s.strip())
        return s

    def to_ewkt(self, trim=False, rounding_precision=-1, output_dimension=3):
        return f'SRID={self.crs.srid};' + self.to_wkt(trim=trim, rounding_precision=rounding_precision, output_dimension=output_dimension)

    def to_geojson(self, keep_crs=False):
        # see https://datatracker.ietf.org/doc/html/rfc7946#section-4
        # convert to WGS lon,lat unless keep_crs is true
        # coords order is always XY

        if keep_crs or self.crs == gws.lib.crs.WGS84:
            return shapely.geometry.mapping(self.geom)

        tr = self.crs.transformer(gws.lib.crs.WGS84)
        new_geom = shapely.ops.transform(tr, self.geom)
        return shapely.geometry.mapping(new_geom)

    def to_precision(self, prec: int):
        geom = shapely.set_precision(self.geom, 10 ** -prec)
        return Shape(geom, self.crs)

    def to_props(self):
        return gws.ShapeProps(crs=self.crs.epsg, geometry=shapely.geometry.mapping(self.geom))

    def is_empty(self):
        return self.geom.is_empty

    def is_ring(self):
        return self.geom.is_ring

    def is_simple(self):
        return self.geom.is_simple

    def is_valid(self):
        return self.geom.is_valid

    def equals(self, other):
        return self._binary_predicate(other, 'equals')

    def contains(self, other):
        return self._binary_predicate(other, 'contains')

    def covers(self, other):
        return self._binary_predicate(other, 'covers')

    def covered_by(self, other):
        return self._binary_predicate(other, 'covered_by')

    def crosses(self, other):
        return self._binary_predicate(other, 'crosses')

    def disjoint(self, other):
        return self._binary_predicate(other, 'disjoint')

    def intersects(self, other):
        return self._binary_predicate(other, 'intersects')

    def overlaps(self, other):
        return self._binary_predicate(other, 'overlaps')

    def touches(self, other):
        return self._binary_predicate(other, 'touches')

    def within(self, other):
        return self._binary_predicate(other, 'within')

    def _binary_predicate(self, other, op):
        """Apply a Shapely predicate to this shape and another one, transformed to this CRS."""
        s = other.transformed_to(self.crs)
        return getattr(self.geom, op)(getattr(s, 'geom'))

    def union(self, others):
        if not others:
            return self

        geoms = [self.geom]
        for s in others:
            s = s.transformed_to(self.crs)
            geoms.append(getattr(s, 'geom'))

        geom = shapely.ops.unary_union(geoms)
        return Shape(geom, self.crs)

    def intersection(self, *others):
        if not others:
            return self

        geom = self.geom
        for s in others:
            s = s.transformed_to(self.crs)
            geom = geom.intersection(getattr(s, 'geom'))

        return Shape(geom, self.crs)

    def to_multi(self):
        if self.type == gws.GeometryType.point:
            return Shape(shapely.geometry.MultiPoint([self.geom]), self.crs)
        if self.type == gws.GeometryType.linestring:
            return Shape(shapely.geometry.MultiLineString([self.geom]), self.crs)
        if self.type == gws.GeometryType.polygon:
            return Shape(shapely.geometry.MultiPolygon([self.geom]), self.crs)
        return self

    def to_type(self, new_type: gws.GeometryType):
        if new_type == self.type:
            return self
        if new_type == gws.GeometryType.geometry:
            return self
        if self.type == gws.GeometryType.point and new_type == gws.GeometryType.multipoint:
            return self.to_multi()
        if self.type == gws.GeometryType.linestring and new_type == gws.GeometryType.multilinestring:
            return self.to_multi()
        if self.type == gws.GeometryType.polygon and new_type == gws.GeometryType.multipolygon:
            return self.to_multi()
        raise Error(f'cannot convert {self.type!r} to {new_type!r}')

    def to_2d(self):
        geom = shapely.force_2d(self.geom)
        if geom is self.geom:
            return self
        return Shape(geom, self.crs)

    def tolerance_polygon(self, tolerance=None, quad_segs=None):
        is_poly = self.type in (gws.GeometryType.polygon, gws.GeometryType.multipolygon)

        if not tolerance and is_poly:
            return self

        # we need a polygon even if tolerance = 0
        tolerance = tolerance or _MIN_TOLERANCE_RADIUS
        quad_segs = quad_segs or _TOLERANCE_QUAD_SEGS

        geom = self.geom
        if self.crs.isGeographic:
            tr = self.crs.transformer(gws.lib.crs.WEBMERCATOR)
            geom = shapely.ops.transform(tr, self.geom)

        if is_poly:
            cs = shapely.geometry.CAP_STYLE.flat
            js = shapely.geometry.JOIN_STYLE.mitre
        else:
            cs = shapely.geometry.CAP_STYLE.round
            js = shapely.geometry.JOIN_STYLE.round

        geom = geom.buffer(tolerance, quad_segs, cap_style=cs, join_style=js)
        if self.crs.isGeographic:
            tr = gws.lib.crs.WEBMERCATOR.transformer(self.crs)
            geom = shapely.ops.transform(tr, geom)

        return Shape(geom, self.crs)

    def transformed_to(self, crs):
        if crs == self.crs:
            return self
        tr = self.crs.transformer(crs)
        dg = shapely.ops.transform(tr, self.geom)
        return Shape(dg, crs)
