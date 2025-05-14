"""GML geometry writer."""

from typing import Optional

import shapely.geometry

import gws
import gws.lib.crs
import gws.lib.extent
import gws.base.feature
import gws.base.shape
import gws.lib.xmlx as xmlx
import gws.lib.uom


# @TODO PostGis options 2 and 4 (https://postgis.net/docs/ST_AsGML.html)

class _Options(gws.Data):
    version: int
    precision: float
    swapxy: bool
    xmlns: str
    crsName: dict


def shape_to_element(
        shape: gws.Shape,
        version: int = 3,
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
        with_xmlns: If ``True`` add the "gml" namespace prefix.
        with_inline_xmlns: If ``True`` add inline "xmlns" attributes.
        namespace: Use this namespace (default "gml").
        crs_format: Crs format to use (default "url" for version 2 and "urn" for version 3).

    Returns:
        A GML element.
    """

    opts = _Options()
    opts.version = int(version)
    if opts.version not in {2, 3}:
        raise gws.Error(f'unsupported GML version {version!r}')

    crs_format = crs_format or (gws.CrsFormat.url if opts.version == 2 else gws.CrsFormat.urn)
    opts.crsName= {'srsName': shape.crs.to_string(crs_format)}

    opts.swapxy = (shape.crs.axis_for_format(crs_format) == gws.Axis.yx) and not always_xy
    opts.precision = coordinate_precision if coordinate_precision is not None else gws.lib.uom.DEFAULT_PRECISION[shape.crs.uom]

    opts.xmlns = ''
    ns = None
    if with_xmlns:
        ns = namespace or xmlx.namespace.require('gml2' if opts.version == 2 else 'gml')
        opts.xmlns = ns.xmlns + ':'

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    fn = _tag2 if opts.version == 2 else _tag3

    # OGC 07-036r1 10.1.4.1
    # If no srsName attribute is given, the CRS shall be specified as part of the larger context this geometry element is part of...
    # NOTE It is expected that the attribute will be specified at the direct position level only in rare cases.

    el = xmlx.tag(*fn(geom, opts))
    if ns and with_inline_xmlns:
        _att_attr(el, 'xmlns:' + ns.xmlns, ns.uri)

    return el


def _point2(geom, opts):
    return f'{opts.xmlns}Point', opts.crsName, _coordinates(geom, opts)


def _point3(geom, opts):
    return f'{opts.xmlns}Point', opts.crsName, _pos(geom, opts)


def _linestring2(geom, opts):
    return f'{opts.xmlns}LineString', opts.crsName, _coordinates(geom, opts)


def _linestring3(geom, opts):
    return [
        f'{opts.xmlns}Curve', opts.crsName,
        [
            f'{opts.xmlns}segments',
            [
                f'{opts.xmlns}LineStringSegment', _pos_list(geom, opts)
            ]
        ]
    ]


def _polygon2(geom, opts):
    return [
        f'{opts.xmlns}Polygon', opts.crsName,
        [
            f'{opts.xmlns}outerBoundaryIs',
            [
                f'{opts.xmlns}LinearRing', _coordinates(geom.exterior, opts)
            ]
        ],
        [
            [
                f'{opts.xmlns}innerBoundaryIs',
                [
                    f'{opts.xmlns}LinearRing', _coordinates(interior, opts)
                ]
            ]
            for interior in geom.interiors
        ]
    ]


def _polygon3(geom, opts):
    return [
        f'{opts.xmlns}Polygon', opts.crsName,
        [
            f'{opts.xmlns}exterior',
            [
                f'{opts.xmlns}LinearRing', _pos_list(geom.exterior, opts)
            ]
        ],
        [
            [
                f'{opts.xmlns}interior',
                [
                    f'{opts.xmlns}LinearRing', _pos_list(interior, opts)
                ]
            ]
            for interior in geom.interiors
        ]
    ]


def _multipoint2(geom, opts):
    return f'{opts.xmlns}MultiPoint', opts.crsName, [[f'{opts.xmlns}pointMember', _tag2(p, opts)] for p in geom.geoms]


def _multipoint3(geom, opts):
    return f'{opts.xmlns}MultiPoint', opts.crsName, [[f'{opts.xmlns}pointMember', _tag3(p, opts)] for p in geom.geoms]


def _multilinestring2(geom, opts):
    return f'{opts.xmlns}MultiLineString', opts.crsName, [[f'{opts.xmlns}lineStringMember', _tag2(p, opts)] for p in geom.geoms]


def _multilinestring3(geom, opts):
    return f'{opts.xmlns}MultiCurve', opts.crsName, [[f'{opts.xmlns}curveMember', _tag3(p, opts)] for p in geom.geoms]


def _multipolygon2(geom, opts):
    return f'{opts.xmlns}MultiPolygon', opts.crsName, [[f'{opts.xmlns}polygonMember', _tag2(p, opts)] for p in geom.geoms]


def _multipolygon3(geom, opts):
    return f'{opts.xmlns}MultiSurface', opts.crsName, [[f'{opts.xmlns}surfaceMember', _tag3(p, opts)] for p in geom.geoms]


def _geometrycollection2(geom, opts):
    return f'{opts.xmlns}MultiGeometry', opts.crsName, [[f'{opts.xmlns}geometryMember', _tag2(p, opts)] for p in geom.geoms]


def _geometrycollection3(geom, opts):
    return f'{opts.xmlns}MultiGeometry', opts.crsName, [[f'{opts.xmlns}geometryMember', _tag3(p, opts)] for p in geom.geoms]


def _pos(geom, opts):
    return f'{opts.xmlns}pos', {'srsDimension': 2}, _pos_list_content(geom, opts)


def _pos_list(geom, opts):
    return f'{opts.xmlns}posList', {'srsDimension': 2}, _pos_list_content(geom, opts)


def _pos_list_content(geom, opts):
    cs = []

    for x, y in geom.coords:
        x = int(x) if opts.precision == 0 else round(x, opts.precision)
        y = int(y) if opts.precision == 0 else round(y, opts.precision)
        if opts.swapxy:
            x, y = y, x
        cs.append(str(x))
        cs.append(str(y))

    return ' '.join(cs)


def _coordinates(geom, opts):
    cs = []

    for x, y in geom.coords:
        x = int(x) if opts.precision == 0 else round(x, opts.precision)
        y = int(y) if opts.precision == 0 else round(y, opts.precision)
        if opts.swapxy:
            x, y = y, x
        cs.append(str(x) + ',' + str(y))

    return f'{opts.xmlns}coordinates', {'decimal': '.', 'cs': ',', 'ts': ' '}, ' '.join(cs)


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


def _tag2(geom, opts):
    typ = geom.geom_type
    fn = _FNS_2.get(typ)
    if fn:
        return fn(geom, opts)
    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')


def _tag3(geom, opts):
    typ = geom.geom_type
    fn = _FNS_3.get(typ)
    if fn:
        return fn(geom, opts)
    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')


def _att_attr(el: gws.XmlElement, key, val):
    el.set(key, val)
    for c in el:
        _att_attr(c, key, val)
