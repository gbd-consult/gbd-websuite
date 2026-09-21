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

    Args:
        shape: A Shape object.
        version: GML version (2 or 3).
        coordinate_precision: The amount of decimal places.
        always_xy: If ``True``, coordinates are assumed to be always in the XY (lon/lat) order.
        with_xmlns: If ``True`` put the elements in the GML namespace.
        with_inline_xmlns: If ``True`` declare the namespace on the geometry element.
        namespace: Use this namespace (default "gml2" for version 2 and "gml3" for version 3).
        crs_format: Crs format to use (default "url" for version 2 and "urn" for version 3).

    Returns:
        A GML element.
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
    version: int
    defaultCrsFormat: gws.CrsFormat
    defaultNamespace: gws.XmlNamespace

    precision: int
    swap_xy: bool
    crsName: dict
    pfx: str
    ns: Optional[gws.XmlNamespace]

    def __init__(self, shape, coordinate_precision, always_xy, with_xmlns, namespace, crs_format):
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
        return tag(self.nsu + name, *args)

    def element(self, geom):
        typ = geom.geom_type
        name = _METHODS.get(typ)
        if name:
            return getattr(self, name)(geom)
        raise gws.Error(f'cannot convert geometry type {typ!r} to GML')

    def multipoint(self, geom):
        return self.t(
            'MultiPoint',
            self.crsName,
            [self.t('pointMember', self.element(p)) for p in geom.geoms],
        )

    def geometrycollection(self, geom):
        return self.t(
            'MultiGeometry',
            self.crsName,
            [self.t('geometryMember', self.element(p)) for p in geom.geoms],
        )

    def round_coords(self, geom):
        for x, y in geom.coords:
            x = int(x) if self.precision == 0 else round(x, self.precision)
            y = int(y) if self.precision == 0 else round(y, self.precision)
            if self.swap_xy:
                x, y = y, x
            yield x, y


class _Writer2(_Writer):
    version = 2
    defaultCrsFormat = gws.CrsFormat.url
    defaultNamespace = xmlx.namespace.c.GML_2

    def point(self, geom):
        return self.t('Point', self.crsName, self.coordinates(geom))

    def linestring(self, geom):
        return self.t('LineString', self.crsName, self.coordinates(geom))

    def polygon(self, geom):
        return self.t(
            'Polygon',
            self.crsName,
            self.t('outerBoundaryIs', self.t('LinearRing', self.coordinates(geom.exterior))),
            [self.t('innerBoundaryIs', self.t('LinearRing', self.coordinates(interior))) for interior in geom.interiors],
        )

    def multilinestring(self, geom):
        return self.t(
            'MultiLineString',
            self.crsName,
            [self.t('lineStringMember', self.element(p)) for p in geom.geoms],
        )

    def multipolygon(self, geom):
        return self.t(
            'MultiPolygon',
            self.crsName,
            [self.t('polygonMember', self.element(p)) for p in geom.geoms],
        )

    def coordinates(self, geom):
        cs = [str(x) + ',' + str(y) for x, y in self.round_coords(geom)]
        return self.t('coordinates', {'decimal': '.', 'cs': ',', 'ts': ' '}, ' '.join(cs))


class _Writer3(_Writer):
    version = 3
    defaultCrsFormat = gws.CrsFormat.urn
    defaultNamespace = xmlx.namespace.c.GML

    def point(self, geom):
        return self.t('Point', self.crsName, self.pos(geom))

    def linestring(self, geom):
        return self.t(
            'Curve',
            self.crsName,
            self.t('segments', self.t('LineStringSegment', self.pos_list(geom))),
        )

    def polygon(self, geom):
        return self.t(
            'Polygon',
            self.crsName,
            self.t('exterior', self.t('LinearRing', self.pos_list(geom.exterior))),
            [self.t('interior', self.t('LinearRing', self.pos_list(interior))) for interior in geom.interiors],
        )

    def multilinestring(self, geom):
        return self.t(
            'MultiCurve',
            self.crsName,
            [self.t('curveMember', self.element(p)) for p in geom.geoms],
        )

    def multipolygon(self, geom):
        return self.t(
            'MultiSurface',
            self.crsName,
            [self.t('surfaceMember', self.element(p)) for p in geom.geoms],
        )

    def pos(self, geom):
        return self.t('pos', {'srsDimension': 2}, self.pos_list_content(geom))

    def pos_list(self, geom):
        return self.t('posList', {'srsDimension': 2}, self.pos_list_content(geom))

    def pos_list_content(self, geom):
        cs = []
        for x, y in self.round_coords(geom):
            cs.append(str(x))
            cs.append(str(y))
        return ' '.join(cs)
