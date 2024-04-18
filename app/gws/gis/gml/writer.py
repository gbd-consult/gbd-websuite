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


# @TODO support GML2
# @TODO PostGis options 2 and 4 (https://postgis.net/docs/ST_AsGML.html)
def shape_to_element(
        shape: gws.Shape,
        coordinate_precision: Optional[int] = None,
        always_xy=False,
        crs_format: gws.CrsFormat = gws.CrsFormat.urn,
        namespace: Optional[gws.XmlNamespace] = None,
        with_xmlns=True,
        with_inline_xmlns=False,
) -> gws.XmlElement:
    """Convert a Shape to a GML3 geometry element."""

    opts = gws.Data()

    opts.axis = gws.Axis.xy if always_xy else shape.crs.axis
    opts.precision = gws.lib.uom.DEFAULT_PRECISION[shape.crs.uom] if coordinate_precision is None else coordinate_precision
    opts.atts = {
        'srsName': shape.crs.to_string(crs_format),
    }

    if with_xmlns:
        ns = namespace or xmlx.namespace.get('gml3')
        opts.ns = ns.xmlns + ':'

    if with_inline_xmlns:
        ns = namespace or xmlx.namespace.get('gml3')
        opts.atts['xmlns:' + ns.xmlns] = ns.uri

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    return xmlx.tag(*_tag(geom, opts))


def _tag(geom, opts):
    typ = geom.geom_type

    if typ == 'Point':
        return f'{opts.ns}Point', opts.atts, _pos(geom, opts, as_list=False)

    if typ == 'LineString':
        return (
            f'{opts.ns}Curve',
            opts.atts,
            f'{opts.ns}segments/{opts.ns}LineStringSegment', opts.atts, _pos(geom, opts)
        )

    if typ == 'Polygon':
        return (
            f'{opts.ns}Polygon',
            opts.atts,
            (
                f'{opts.ns}exterior/{opts.ns}LinearRing', opts.atts, _pos(geom.exterior, opts)),
            [
                (f'{opts.ns}interior/{opts.ns}LinearRing', opts.atts, _pos(interior, opts))
                for interior in geom.interiors
            ]
        )

    if typ == 'MultiPoint':
        return f'{opts.ns}MultiPoint', opts.atts, [(f'{opts.ns}pointMember', _tag(p, opts)) for p in geom.geoms]

    if typ == 'MultiLineString':
        return f'{opts.ns}MultiCurve', opts.atts, [(f'{opts.ns}curveMember', _tag(p, opts)) for p in geom.geoms]

    if typ == 'MultiPolygon':
        return f'{opts.ns}MultiSurface', opts.atts, [(f'{opts.ns}surfaceMember', _tag(p, opts)) for p in geom.geoms]

    raise gws.Error(f'cannot convert geometry type {typ!r} to GML')


def _pos(geom, opts, as_list=True):
    cs = []

    if opts.axis == gws.Axis.xy:
        for x, y in geom.coords:
            cs.append(x)
            cs.append(y)
    else:
        for x, y in geom.coords:
            cs.append(y)
            cs.append(x)

    cs = [round(c, opts.precision) for c in cs]
    tag = 'posList' if as_list else 'pos'
    return f'{opts.ns}{tag}', {'srsDimension': 2}, ' '.join(str(c) for c in cs)
