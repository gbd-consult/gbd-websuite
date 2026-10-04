"""GML geometry writer."""

from typing import Optional

import shapely.geometry

import gws
import gws.lib.uom
import gws.lib.xmlx as xmlx
from gws.lib.xmlx import tag

# @TODO PostGis options 2 and 4 (https://postgis.net/docs/ST_AsGML.html)


DEFAULT_VERSION = 3


def shape_to_element(
    shape: gws.Shape,
    version: int = DEFAULT_VERSION,
    coordinate_precision: Optional[int] = None,
    always_xy: bool = False,
    with_xmlns: bool = True,
    with_inline_xmlns: bool = False,
    namespace: Optional[gws.XmlNamespace] = None,
    crs_format: Optional[gws.CrsFormat] = None,
) -> gws.XmlElement:
    """Convert a Shape to a GML geometry element.

    In GML 3, line strings are written as ``Curve`` elements and multi line strings
    and multi polygons as ``MultiCurve`` and ``MultiSurface``.

    Args:
        shape: A Shape object.
        version: GML version (2 or 3).
        coordinate_precision: Number of decimal places. The default depends on the units of the shape CRS.
        always_xy: If ``True``, coordinates are always written in the XY (lon/lat) order,
            otherwise in the axis order of the CRS.
        with_xmlns: If ``True``, put the elements in the GML namespace.
        with_inline_xmlns: If ``True``, declare the namespace on the geometry element.
        namespace: Namespace to use (default ``GML_2`` for version 2 and ``GML``, that is GML 3.2, for version 3).
        crs_format: CRS format for ``srsName`` (default ``url`` for version 2 and ``urn`` for version 3).

    Returns:
        A GML element.

    Raises:
        ``gws.Error``: If the version or the geometry type is not supported.
    """

    v = int(version or DEFAULT_VERSION)
    if v == 2:
        cls = _Writer2
    elif v == 3:
        cls = _Writer3
    else:
        raise gws.Error(f'unsupported GML version {version!r}')

    wr = cls(shape, coordinate_precision, always_xy, with_xmlns, namespace, crs_format)

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')

    # OGC 07-036r1 10.1.4.1
    # If no srsName attribute is given, the CRS shall be specified as part of the larger context this geometry element is part of...
    # NOTE It is expected that the attribute will be specified at the direct position level only in rare cases.

    el = wr.element(geom)
    if wr.ns and with_inline_xmlns:
        el.declare(wr.ns)

    return el


_METHODS = {
    'Point': 'point',
    'LineString': 'linestring',
    'Polygon': 'polygon',
    'MultiPoint': 'multipoint',
    'MultiLineString': 'multilinestring',
    'MultiPolygon': 'multipolygon',
    'GeometryCollection': 'geometrycollection',
}


class _Writer:
    """Base GML writer, holds the output options and the version-independent geometry types."""

    version: int
    """GML version."""
    defaultCrsFormat: gws.CrsFormat
    """CRS format used when none is given."""
    defaultNamespace: gws.XmlNamespace
    """Namespace used when none is given."""

    precision: int
    """Number of decimal places for coordinates."""
    swap_xy: bool
    """If ``True``, coordinates are written in the YX order."""
    crsName: dict
    """The ``srsName`` attribute for geometry elements."""
    pfx: str
    """Not used."""
    ns: Optional[gws.XmlNamespace]
    """Namespace of the elements, or ``None`` for no namespace."""

    def __init__(self, shape, coordinate_precision, always_xy, with_xmlns, namespace, crs_format):
        """Create a writer.

        Args:
            shape: The shape to write, provides the CRS.
            coordinate_precision: Number of decimal places, or ``None`` for the default of the CRS units.
            always_xy: If ``True``, write coordinates in the XY order.
            with_xmlns: If ``True``, put the elements in a namespace.
            namespace: Namespace to use, or ``None`` for the default.
            crs_format: CRS format to use, or ``None`` for the default.
        """

        crs_format = crs_format or self.defaultCrsFormat
        self.crsName = {'srsName': shape.crs.to_string(crs_format)}

        self.swap_xy = (shape.crs.axis_for_format(crs_format) == gws.Axis.yx) and not always_xy
        
        self.precision = gws.lib.uom.DEFAULT_PRECISION[shape.crs.uom]
        if coordinate_precision is not None:
            self.precision = coordinate_precision

        self.ns = None
        self.nsu = ''
        if with_xmlns:
            self.ns = namespace or self.defaultNamespace
            self.nsu = '{' + self.ns.uri + '}'

    def t(self, name, *args):
        """Create an element in the writer namespace.

        Args:
            name: Element name without a namespace.
            *args: Attributes, text and child elements, as accepted by ``gws.lib.xmlx.tag``.

        Returns:
            An XML element.
        """

        return tag(self.nsu + name, *args)

    def element(self, geom):
        """Convert a geometry to a GML element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.

        Raises:
            ``gws.Error``: If the geometry type is not supported.
        """

        typ = geom.geom_type
        name = _METHODS.get(typ)
        if name:
            return getattr(self, name)(geom)
        raise gws.Error(f'cannot convert geometry type {typ!r} to GML')

    def multipoint(self, geom):
        """Create a ``MultiPoint`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'MultiPoint',
            self.crsName,
            [self.t('pointMember', self.element(p)) for p in geom.geoms],
        )

    def geometrycollection(self, geom):
        """Create a ``MultiGeometry`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'MultiGeometry',
            self.crsName,
            [self.t('geometryMember', self.element(p)) for p in geom.geoms],
        )

    def round_coords(self, geom):
        """Round the coordinates of a geometry and swap them if needed.

        Args:
            geom: A shapely geometry with a ``coords`` sequence.

        Yields:
            Coordinate pairs in the output axis order.
        """

        for x, y in geom.coords:
            x = int(x) if self.precision == 0 else round(x, self.precision)
            y = int(y) if self.precision == 0 else round(y, self.precision)
            if self.swap_xy:
                x, y = y, x
            yield x, y


class _Writer2(_Writer):
    """GML 2 writer."""

    version = 2
    defaultCrsFormat = gws.CrsFormat.url
    defaultNamespace = xmlx.namespace.c.GML_2

    def point(self, geom):
        """Create a ``Point`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t('Point', self.crsName, self.coordinates(geom))

    def linestring(self, geom):
        """Create a ``LineString`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t('LineString', self.crsName, self.coordinates(geom))

    def polygon(self, geom):
        """Create a ``Polygon`` element with ``outerBoundaryIs`` and ``innerBoundaryIs`` rings.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'Polygon',
            self.crsName,
            self.t('outerBoundaryIs', self.t('LinearRing', self.coordinates(geom.exterior))),
            [self.t('innerBoundaryIs', self.t('LinearRing', self.coordinates(interior))) for interior in geom.interiors],
        )

    def multilinestring(self, geom):
        """Create a ``MultiLineString`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'MultiLineString',
            self.crsName,
            [self.t('lineStringMember', self.element(p)) for p in geom.geoms],
        )

    def multipolygon(self, geom):
        """Create a ``MultiPolygon`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'MultiPolygon',
            self.crsName,
            [self.t('polygonMember', self.element(p)) for p in geom.geoms],
        )

    def coordinates(self, geom):
        """Create a ``coordinates`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        cs = [str(x) + ',' + str(y) for x, y in self.round_coords(geom)]
        return self.t('coordinates', {'decimal': '.', 'cs': ',', 'ts': ' '}, ' '.join(cs))


class _Writer3(_Writer):
    """GML 3 writer."""

    version = 3
    defaultCrsFormat = gws.CrsFormat.urn
    defaultNamespace = xmlx.namespace.c.GML

    def point(self, geom):
        """Create a ``Point`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t('Point', self.crsName, self.pos(geom))

    def linestring(self, geom):
        """Create a ``Curve`` element with a single ``LineStringSegment``.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'Curve',
            self.crsName,
            self.t('segments', self.t('LineStringSegment', self.pos_list(geom))),
        )

    def polygon(self, geom):
        """Create a ``Polygon`` element with ``exterior`` and ``interior`` rings.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'Polygon',
            self.crsName,
            self.t('exterior', self.t('LinearRing', self.pos_list(geom.exterior))),
            [self.t('interior', self.t('LinearRing', self.pos_list(interior))) for interior in geom.interiors],
        )

    def multilinestring(self, geom):
        """Create a ``MultiCurve`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'MultiCurve',
            self.crsName,
            [self.t('curveMember', self.element(p)) for p in geom.geoms],
        )

    def multipolygon(self, geom):
        """Create a ``MultiSurface`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t(
            'MultiSurface',
            self.crsName,
            [self.t('surfaceMember', self.element(p)) for p in geom.geoms],
        )

    def pos(self, geom):
        """Create a ``pos`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t('pos', {'srsDimension': 2}, self.pos_list_content(geom))

    def pos_list(self, geom):
        """Create a ``posList`` element.

        Args:
            geom: A shapely geometry.

        Returns:
            A GML element.
        """

        return self.t('posList', {'srsDimension': 2}, self.pos_list_content(geom))

    def pos_list_content(self, geom):
        """Format the coordinates of a geometry as a space-separated list.

        Args:
            geom: A shapely geometry with a ``coords`` sequence.

        Returns:
            Coordinates as a string.
        """

        cs = []
        for x, y in self.round_coords(geom):
            cs.append(str(x))
            cs.append(str(y))
        return ' '.join(cs)
