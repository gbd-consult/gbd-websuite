import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.uom as units
import gws.lib.xmlx as xmlx
import gws.types as t

from .. import core
from .. import parseutil as u


# http://portal.opengeospatial.org/files/?artifact_id=35326


def parse(xml: str) -> core.Caps:
    caps_el = xmlx.from_string(xml, compact_whitespace=True, remove_namespaces=True)
    tms_lst = [_tile_matrix_set(el) for el in caps_el.findall('Contents/TileMatrixSet')]
    tms_dct = {tms.uid: tms for tms in tms_lst}
    sls = gws.gis.source.check_layers(
        _layer(el, tms_dct) for el in caps_el.findall('Contents/Layer'))
    return core.Caps(
        tileMatrixSets=tms_lst,
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        sourceLayers=sls,
        version=caps_el.get('version'))


def _layer(layer_el: gws.IXmlElement, tms_dct):
    # <Layer>
    #   <ows:Title>...
    #   <Style>...
    #   <Format>...
    #   <TileMatrixSetLink>
    #     <TileMatrixSet>...

    sl = gws.SourceLayer()

    sl.metadata = u.element_metadata(layer_el)
    sl.name = sl.metadata.get('name', '')
    sl.title = sl.metadata.get('title', '')

    sl.styles = [u.parse_style(e) for e in layer_el.findall('Style')]
    sl.defaultStyle = u.default_style(sl.styles)
    if sl.defaultStyle:
        sl.legendUrl = sl.defaultStyle.legendUrl

    sl.tileMatrixIds = [el.textof('TileMatrixSet') for el in layer_el.findall('TileMatrixSetLink')]
    sl.tileMatrixSets = [tms_dct[tid] for tid in sl.tileMatrixIds]

    extra_crsids = [tms.crs.srid for tms in sl.tileMatrixSets]
    wgs_bounds = u.wgs_bounds(layer_el)
    crs_list = u.supported_crs(layer_el, extra_crsids)

    sl.supportedCrs = crs_list or [gws.gis.crs.WGS84]
    sl.wgsExtent = wgs_bounds.extent if wgs_bounds else gws.gis.crs.WGS84.wgsExtent

    sl.isImage = True
    sl.isVisible = True

    sl.imageFormat = layer_el.textof('Format')

    sl.resourceUrls = {
        e.get('resourceType'): e.get('template')
        for e in layer_el.findall('ResourceURL')
    }

    return sl


def _tile_matrix_set(tms_el: gws.IXmlElement):
    # <TileMatrixSet>
    #   <ows:Identifier>...
    #   <ows:SupportedCRS>...
    #   <TileMatrix>
    #     ...

    tms = gws.TileMatrixSet()

    tms.uid = tms_el.textof('Identifier')
    tms.crs = gws.gis.crs.require(tms_el.textof('SupportedCRS'))
    tms.matrices = sorted(
        [_tile_matrix(e) for e in tms_el.findall('TileMatrix')],
        key=lambda m: -m.scale)

    return tms


def _tile_matrix(tm_el: gws.IXmlElement):
    # <TileMatrix>
    #   <ows:Identifier>
    #   <ScaleDenominator>
    #   ...

    tm = gws.TileMatrix()
    tm.uid = tm_el.textof('Identifier')
    tm.scale = u.to_float(tm_el.textof('ScaleDenominator'))

    p = u.to_float_pair(tm_el.textof('TopLeftCorner'))
    tm.x = p[0]
    tm.y = p[1]

    tm.width = u.to_int(tm_el.textof('MatrixWidth'))
    tm.height = u.to_int(tm_el.textof('MatrixHeight'))

    tm.tileWidth = u.to_int(tm_el.textof('TileWidth'))
    tm.tileHeight = u.to_int(tm_el.textof('TileHeight'))

    tm.extent = _extent_for_matrix(tm)

    return tm


# compute a bbox for a TileMatrix
# see http://portal.opengeospatial.org/files/?artifact_id=35326 page 8

def _extent_for_matrix(m: gws.TileMatrix):
    res = units.scale_to_res(m.scale)

    return [
        m.x,
        m.y - res * m.height * m.tileHeight,
        m.x + res * m.width * m.tileWidth,
        m.y,
    ]
