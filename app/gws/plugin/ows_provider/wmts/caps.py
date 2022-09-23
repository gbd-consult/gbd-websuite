import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.units as units
import gws.lib.xmlx as xmlx
import gws.types as t

from .. import core
from .. import parseutil as u


# http://portal.opengeospatial.org/files/?artifact_id=35326


class Caps(core.Caps):
    tile_matrix_sets: t.List[gws.TileMatrixSet]


def parse(xml):
    caps_el = xmlx.from_string(xml, compact_whitespace=True, remove_namespaces=True)
    tile_matrix_sets = [_tile_matrix_set(el) for el in caps_el.findall('Contents/TileMatrixSet')]
    tms_map = {tms.uid: tms for tms in tile_matrix_sets}
    source_layers = gws.gis.source.check_layers(
        _layer(el, tms_map) for el in caps_el.findall('Contents/Layer'))
    return Caps(
        tile_matrix_sets=tile_matrix_sets,
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        source_layers=source_layers,
        version=caps_el.get('version'))


def _layer(layer_el: gws.IXmlElement, tms_map):
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
    sl.default_style = u.default_style(sl.styles)
    if sl.default_style:
        sl.legend_url = sl.default_style.legend_url

    sl.tile_matrix_ids = [el.text_of('TileMatrixSet') for el in layer_el.findall('TileMatrixSetLink')]
    sl.tile_matrix_sets = [tms_map[tid] for tid in sl.tile_matrix_ids]

    extra_crsids = [tms.crs.srid for tms in sl.tile_matrix_sets]
    sl.supported_bounds = u.supported_bounds(layer_el, extra_crsids)

    sl.is_image = True
    sl.is_visible = True

    sl.image_format = layer_el.text_of('Format')

    sl.resource_urls = {
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

    tms.uid = tms_el.text_of('Identifier')
    tms.crs = gws.gis.crs.require(tms_el.text_of('SupportedCRS'))
    tms.matrices = sorted(
        [_tile_matrix(e) for e in tms_el.findall('TileMatrix')],
        key=lambda m: int('1' + m.uid))

    return tms


def _tile_matrix(tm_el: gws.IXmlElement):
    # <TileMatrix>
    #   <ows:Identifier>
    #   <ScaleDenominator>
    #   ...

    tm = gws.TileMatrix()
    tm.uid = tm_el.text_of('Identifier')
    tm.scale = u.to_float(tm_el.text_of('ScaleDenominator'))

    p = u.to_float_pair(tm_el.text_of('TopLeftCorner'))
    tm.x = p[0]
    tm.y = p[1]

    tm.width = u.to_int(tm_el.text_of('MatrixWidth'))
    tm.height = u.to_int(tm_el.text_of('MatrixHeight'))

    tm.tile_width = u.to_int(tm_el.text_of('TileWidth'))
    tm.tile_height = u.to_int(tm_el.text_of('TileHeight'))

    tm.extent = _extent_for_matrix(tm)

    return tm


# compute a bbox for a TileMatrix
# see http://portal.opengeospatial.org/files/?artifact_id=35326 page 8

def _extent_for_matrix(m: gws.TileMatrix):
    res = units.scale_to_res(m.scale)

    return [
        m.x,
        m.y - res * m.height * m.tile_height,
        m.x + res * m.width * m.tile_width,
        m.y,
    ]
