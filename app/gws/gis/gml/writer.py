"""GML geometry writer."""

from typing import Optional

import shapely.geometry

import gws
import gws.gis.crs
import gws.gis.extent
import gws.base.feature
import gws.base.shape
import gws.lib.xmlx as xmlx
import gws.lib.uom


# @TODO PostGis options 2 and 4 (https://postgis.net/docs/ST_AsGML.html)

def shape_to_element(
        shape: gws.Shape,
        version: int = 3,
        coordinate_precision: Optional[int] = None,
        always_xy: bool = False,
        with_xmlns: bool = True,
        with_inline_xmlns: bool = False,
) -> gws.XmlElement:
    """Convert a Shape to a GML geometry element.

    Args:
        shape: A Shape object.
        version: GML version (2 or 3)
        coordinate_precision: The amount of decimal places.
        always_xy: If ``True``, coordinates are assumed to be always in the XY (lon/lat) order.
        with_xmlns: If ``True`` add the "gml" namespace prefix.
        with_inline_xmlns: If ``True`` add inline "xmlns" attributes.

    Returns:
        A GML element.
    """

    opts = gws.Data()
    opts.version = int(version)
    if opts.version not in {2, 3}:
        raise gws.Error(f'unsupported GML version {version!r}')

    crs_format = gws.CrsFormat.url if opts.version == 2 else gws.CrsFormat.urn
    crs = shape.crs.to_string(crs_format)

    opts.swapxy = (shape.crs.axis_for_format(crs_format) == gws.Axis.yx) and not always_xy

    opts.precision = coordinate_precision
    if opts.precision is None:
        opts.precision = gws.lib.uom.DEFAULT_PRECISION[shape.crs.uom]

    opts.ns = ''
    opts.xmlns = None

    if with_xmlns:
        ns = xmlx.namespace.get('gml2' if opts.version == 2 else 'gml3')
        opts.ns = ns.xmlns + ':'
        if with_inline_xmlns:
            opts.xmlns = {'xmlns:' + ns.xmlns: ns.uri}

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    fn = _tag2 if opts.version == 2 else _tag3

    # OGC 07-036r1 10.1.4.1
    # If no srsName attribute is given, the CRS shall be specified as part of the larger context this geometry element is part of...
    # NOTE It is expected that the attribute will be specified at the direct position level only in rare cases.

    return xmlx.tag(*fn(geom, opts, {'srsName': crs}))


def _point2(geom, opts, crs):
    return f'{opts.ns}Point', opts.xmlns, crs, _coordinates(geom, opts)


def _point3(geom, opts, crs):
    return f'{opts.ns}Point', opts.xmlns, crs, _pos(geom, opts)


def _linestring2(geom, opts, crs):
    return f'{opts.ns}LineString', opts.xmlns, crs, _coordinates(geom, opts)


def _linestring3(geom, opts, crs):
    return [
        f'{opts.ns}Curve', opts.xmlns, crs,
        [
            f'{opts.ns}segments', opts.xmlns,
            [
                f'{opts.ns}LineStringSegment', opts.xmlns, _pos_list(geom, opts)
            ]
        ]
    ]


def _polygon2(geom, opts, crs):
    return [
        f'{opts.ns}Polygon', opts.xmlns, crs,
        [
            f'{opts.ns}outerBoundaryIs', opts.xmlns,
            [
                f'{opts.ns}LinearRing', opts.xmlns, _coordinates(geom.exterior, opts)
            ]
        ],
        [
            [
                f'{opts.ns}innerBoundaryIs', opts.xmlns,
                [
                    f'{opts.ns}LinearRing', opts.xmlns, _coordinates(interior, opts)
                ]
            ]
            for interior in geom.interiors
        ]
    ]


def _polygon3(geom, opts, crs):
    return [
        f'{opts.ns}Polygon', opts.xmlns, crs,
        [
            f'{opts.ns}exterior', opts.xmlns,
            [
                f'{opts.ns}LinearRing', opts.xmlns, _pos_list(geom.exterior, opts)
            ]
        ],
        [
            [
                f'{opts.ns}interior', opts.xmlns,
                [
                    f'{opts.ns}LinearRing', opts.xmlns, _pos_list(interior, opts)
                ]
            ]
            for interior in geom.interiors
        ]
    ]


def _multipoint2(geom, opts, crs):
    return f'{opts.ns}MultiPoint', opts.xmlns, crs, [[f'{opts.ns}pointMember', opts.xmlns, _tag2(p, opts)] for p in geom.geoms]


def _multipoint3(geom, opts, crs):
    return f'{opts.ns}MultiPoint', opts.xmlns, crs, [[f'{opts.ns}pointMember', opts.xmlns, _tag3(p, opts)] for p in geom.geoms]


def _multilinestring2(geom, opts, crs):
    return f'{opts.ns}MultiLineString', opts.xmlns, crs, [[f'{opts.ns}lineStringMember', opts.xmlns, _tag2(p, opts)] for p in geom.geoms]


def _multilinestring3(geom, opts, crs):
    return f'{opts.ns}MultiCurve', opts.xmlns, crs, [[f'{opts.ns}curveMember', opts.xmlns, _tag3(p, opts)] for p in geom.geoms]


def _multipolygon2(geom, opts, crs):
    return f'{opts.ns}MultiPolygon', opts.xmlns, crs, [[f'{opts.ns}polygonMember', opts.xmlns, _tag2(p, opts)] for p in geom.geoms]


def _multipolygon3(geom, opts, crs):
    return f'{opts.ns}MultiSurface', opts.xmlns, crs, [[f'{opts.ns}surfaceMember', opts.xmlns, _tag3(p, opts)] for p in geom.geoms]


def _geometrycollection2(geom, opts, crs):
    return f'{opts.ns}MultiGeometry', opts.xmlns, crs, [[f'{opts.ns}geometryMember', opts.xmlns, _tag2(p, opts)] for p in geom.geoms]


def _geometrycollection3(geom, opts, crs):
    return f'{opts.ns}MultiGeometry', opts.xmlns, crs, [[f'{opts.ns}geometryMember', opts.xmlns, _tag3(p, opts)] for p in geom.geoms]


def _pos(geom, opts):
    return f'{opts.ns}pos', opts.xmlns, {'srsDimension': 2}, _pos_list_content(geom, opts)


def _pos_list(geom, opts):
    return f'{opts.ns}posList', opts.xmlns, {'srsDimension': 2}, _pos_list_content(geom, opts)


def _pos_list_content(geom, opts):
    cs = []

    for x, y in geom.coords:
        x = int(x) if opts.precision == 0 else round(x, opts.precision)
        y = int(y) if opts.precision == 0 else round(y, opts.precision)
        if opts.swapxy:
            x, y = y, x
        cs.append(x)
        cs.append(y)

    return ' '.join(str(c) for c in cs)


def _coordinates(geom, opts):
    cs = []

    for x, y in geom.coords:
        x = int(x) if opts.precision == 0 else round(x, opts.precision)
        y = int(y) if opts.precision == 0 else round(y, opts.precision)
        if opts.swapxy:
            x, y = y, x
        cs.append(str(x) + ',' + str(y))

    return f'{opts.ns}coordinates', opts.xmlns, {'decimal': '.', 'cs': ',', 'ts': ' '}, ' '.join(cs)


_FNS_2 = {
    'Point': _point2,
    'LineString': _linestring2,
    'Polygon': _polygon2,
    'MultiPoint': _multipoint2,
    'MultiLineString': _multilinestring2,
    'MultiPolygon': _multipolygon2,
    'GeometryCollection': _geometrycollection2,
}

_FNS_3 = {
    'Point': _point3,
    'LineString': _linestring3,
    'Polygon': _polygon3,
    'MultiPoint': _multipoint3,
    'MultiLineString': _multilinestring3,
    'MultiPolygon': _multipolygon3,
    'GeometryCollection': _geometrycollection3,
}


def _tag2(geom, opts, crs=None):
    typ = geom.geom_type
    fn = _FNS_2.get(typ)
    if fn:
        return fn(geom, opts, crs)
    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')


def _tag3(geom, opts, crs=None):
    typ = geom.geom_type
    fn = _FNS_3.get(typ)
    if fn:
        return fn(geom, opts, crs)
    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')
