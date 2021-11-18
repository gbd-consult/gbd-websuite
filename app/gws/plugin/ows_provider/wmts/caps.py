import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.units as units
import gws.lib.xml3 as xml3
import gws.types as t

from .. import core
from .. import parseutil as u


# http://portal.opengeospatial.org/files/?artifact_id=35326


class Caps(core.Caps):
    tile_matrix_sets: t.List[gws.TileMatrixSet]


def parse(xml):
    root_el = xml3.from_string(xml, compact_ws=True, strip_ns=True)
    tile_matrix_sets = [_tile_matrix_set(e) for e in xml3.all(root_el, 'Contents.TileMatrixSet')]
    tms_map = {tms.uid: tms for tms in tile_matrix_sets}
    source_layers = gws.gis.source.check_layers(
        _layer(e, tms_map) for e in xml3.all(root_el, 'Contents.Layer'))
    return Caps(
        tile_matrix_sets=tile_matrix_sets,
        metadata=u.service_metadata(root_el),
        operations=u.service_operations(root_el),
        source_layers=source_layers,
        version=xml3.attr(root_el, 'version'))


def _layer(el: gws.XmlElement, tms_map):
    # <Layer>
    #   <ows:Title>...
    #   <Style>...
    #   <Format>...
    #   <TileMatrixSetLink>
    #     <TileMatrixSet>...

    sl = gws.SourceLayer()

    sl.metadata = u.element_metadata(el)
    sl.name = sl.metadata.get('name', '')
    sl.title = sl.metadata.get('title', '')

    sl.styles = [u.parse_style(e) for e in xml3.all(el, 'Style')]
    sl.default_style = u.default_style(sl.styles)
    if sl.default_style:
        sl.legend_url = sl.default_style.legend_url

    sl.tile_matrix_ids = [xml3.text(e, 'TileMatrixSet') for e in xml3.all(el, 'TileMatrixSetLink')]
    sl.tile_matrix_sets = [tms_map[tid] for tid in sl.tile_matrix_ids]

    extra_crsids = [tms.crs.srid for tms in sl.tile_matrix_sets]
    sl.supported_bounds = u.supported_bounds(el, extra_crsids)

    sl.is_image = True
    sl.is_visible = True

    sl.image_format = xml3.text(el, 'Format')

    sl.resource_urls = {
        xml3.attr(e, 'resourceType'): xml3.attr(e, 'template')
        for e in xml3.all(el, 'ResourceURL')
    }

    return sl


def _tile_matrix_set(el: gws.XmlElement):
    # <TileMatrixSet>
    #   <ows:Identifier>...
    #   <ows:SupportedCRS>...
    #   <TileMatrix>
    #     ...
    
    tms = gws.TileMatrixSet()

    tms.uid = xml3.text(el, 'Identifier')
    tms.crs = gws.gis.crs.require(xml3.text(el, 'SupportedCRS'))
    tms.matrices = sorted(
        [_tile_matrix(e) for e in xml3.all(el, 'TileMatrix')],
        key=lambda m: int('1' + m.uid))

    return tms


def _tile_matrix(el: gws.XmlElement):
    # <TileMatrix>
    #   <ows:Identifier>
    #   <ScaleDenominator>
    #   ...
    
    tm = gws.TileMatrix()
    tm.uid = xml3.text(el, 'Identifier')
    tm.scale = u.to_float(xml3.text(el, 'ScaleDenominator'))

    p = u.to_float_pair(xml3.text(el, 'TopLeftCorner'))
    tm.x = p[0]
    tm.y = p[1]

    tm.width = u.to_int(xml3.text(el, 'MatrixWidth'))
    tm.height = u.to_int(xml3.text(el, 'MatrixHeight'))

    tm.tile_width = u.to_int(xml3.text(el, 'TileWidth'))
    tm.tile_height = u.to_int(xml3.text(el, 'TileHeight'))

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
