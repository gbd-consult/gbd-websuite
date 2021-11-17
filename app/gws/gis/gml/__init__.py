import osgeo.gdal
import shapely.geometry

import gws
import gws.gis.crs
import gws.gis.extent
import gws.gis.feature
import gws.gis.shape
import gws.lib.xml2
import gws.lib.xml3 as xml3
import gws.types as t


def tag(*args):
    return args


def parse_envelope(el: gws.lib.xml2.Element) -> t.Optional[gws.Bounds]:
    """Parse a gml:Envelope.

    See:
        OGC 07-036, 10.1.4.6 EnvelopeType, Envelope
    """
    # @TODO "coordinates" and "pos" are deprecated, but still should be parsed

    if not el:
        return None

    crs = gws.gis.crs.get(el.attr('srsName') or 4326)
    if not crs:
        return None

    def pair(s):
        p = s.split()
        return float(p[0]), float(p[1])

    try:
        x1, y1 = pair(el.get_text('LowerCorner'))
        x2, y2 = pair(el.get_text('UpperCorner'))
        ext = gws.gis.extent.from_list([x1, y1, x2, y2])
        return gws.Bounds(crs=crs, extent=ext)
    except (IndexError, TypeError):
        pass


def shape_to_element(
    shape: gws.IShape,
    precision=0,
    invert_axis=False,
    crs_format: gws.CrsFormat = gws.CrsFormat.URN
) -> gws.XmlElement:

    def pos(geom, as_list=True):
        cs = []

        if invert_axis:
            for x, y in geom.coords:
                cs.append(y)
                cs.append(x)
        else:
            for x, y in geom.coords:
                cs.append(x)
                cs.append(y)

        if precision:
            cs = [round(c, precision) for c in cs]
        else:
            cs = [int(c) for c in cs]

        return (
            'gml:posList' if as_list else 'gml:pos',
            {'srsDimension': 2},
            ' '.join(str(c) for c in cs))

    def convert(geom, srs=None):
        typ = geom.type

        if typ == 'Point':
            return 'gml:Point', srs, pos(geom, False)

        if typ == 'LineString':
            return 'gml:LineString', srs, pos(geom)

        if typ == 'Polygon':
            return (
                'gml:Polygon',
                srs,
                ('gml:exterior gml:LinearRing', pos(geom.exterior)),
                [('gml:interior gml:LinearRing', pos(p)) for p in geom.interiors]
            )

        if typ == 'MultiPoint':
            return 'gml:MultiPoint', srs, [('gml:pointMember', convert(p)) for p in geom]

        if typ == 'MultiLineString':
            return 'gml:MultiCurve', srs, [('gml:curveMember', convert(p)) for p in geom]

        if typ == 'MultiPolygon':
            return 'gml:MultiSurface', srs, [('gml:surfaceMember', convert(p)) for p in geom]

    geom: shapely.geometry.base.BaseGeometry = getattr(shape, 'geom')
    srs = shape.crs.to_string(crs_format)
    return xml3.tag(*convert(geom, {'srsName': srs}))


def features_from_xml(xml, invert_axis=False):
    tmp = '/vsimem/' + gws.random_string(64) + '.xml'
    osgeo.gdal.FileFromMemBuffer(tmp, xml)

    ds = osgeo.gdal.OpenEx(
        tmp,
        allowed_drivers=['gml'],
        open_options=[
            'SWAP_COORDINATES=%s' % ('YES' if invert_axis else 'NO'),
            'DOWNLOAD_SCHEMA=NO'
        ])

    if not ds:
        gws.log.error('gdal.gml driver failed')
        osgeo.gdal.Unlink(tmp)
        return None

    fs = list(_features_from_gdal(ds))
    ds = None
    osgeo.gdal.Unlink(tmp)

    return fs


def _features_from_gdal(ds):
    for n in range(ds.GetLayerCount()):
        layer = ds.GetLayer(n)
        layer_name = layer.GetName()

        for feature in layer:
            atts = {
                feature.GetFieldDefnRef(n).GetName(): feature.GetFieldAsString(n)
                for n in range(feature.GetFieldCount())
            }

            # GetFID returns a number, which is mostly not the case
            uid = feature.GetFID()
            if not uid and 'fid' in atts:
                uid = atts.pop('fid')

            yield gws.gis.feature.Feature(
                uid=uid,
                category=layer_name,
                shape=_shape_from_gdal(feature.GetGeomFieldRef(0)),
                attributes=atts
            )


def _shape_from_gdal(geom):
    if not geom:
        return
    sr = geom.GetSpatialReference()
    if not sr:
        return
    crs = gws.gis.crs.get(sr.GetAuthorityCode(None))
    wkt = geom.ExportToWkt()
    return gws.gis.shape.from_wkt(wkt, crs)
