import gws
import gws.gis.crs
import gws.gis.ows.parseutil as u
import gws.lib.units as units
import gws.lib.xml2
import gws.types as t

from .. import core

# http://portal.opengeospatial.org/files/?artifact_id=35326


class Caps(core.Caps):
    tile_matrix_sets: t.List[gws.TileMatrixSet]


def parse(xml):
    root_el = gws.lib.xml2.from_string(xml)

    tile_matrix_sets = [_tile_matrix_set(e) for e in root_el.all('Contents.TileMatrixSet')]
    tms_map = {tms.uid: tms for tms in tile_matrix_sets}

    source_layers = u.enum_source_layers(_layer(e, tms_map) for e in root_el.all('Contents.Layer'))

    return Caps(
        tile_matrix_sets=tile_matrix_sets,
        metadata=u.get_service_metadata(root_el),
        operations=u.get_operations(root_el),
        source_layers=source_layers,
        version=root_el.attr('version'),
    )


def _layer(el, tms_map):
    sl = gws.SourceLayer()

    sl.metadata = u.get_metadata(el)
    sl.name = sl.metadata.get('name', '')
    sl.title = sl.metadata.get('title', '')

    sl.styles = [u.get_style(e) for e in el.all('Style')]
    sl.default_style = u.default_style(sl.styles)
    if sl.default_style:
        sl.legend_url = sl.default_style.legend_url

    sl.tile_matrix_ids = [e.get_text('TileMatrixSet') for e in el.all('TileMatrixSetLink')]
    sl.tile_matrix_sets = [tms_map[tid] for tid in sl.tile_matrix_ids]

    extra_crsids = set(tms.crs.srid for tms in sl.tile_matrix_sets)
    sl.supported_bounds = u.get_supported_bounds(el, extra_crsids)

    sl.is_image = True
    sl.is_visible = True

    sl.image_format = el.get_text('Format')

    sl.resource_urls = {
        rs.attr('resourceType'): rs.attr('template')
        for rs in el.all('ResourceURL')
    }

    return sl


def _tile_matrix_set(el):
    tms = gws.TileMatrixSet()

    tms.uid = el.get_text('Identifier')
    tms.crs = gws.gis.crs.require(el.get_text('SupportedCRS'))
    tms.matrices = sorted(
        [_tile_matrix(e) for e in el.all('TileMatrix')],
        key=lambda m: int('1' + m.uid))

    return tms


def _tile_matrix(el):
    tm = gws.TileMatrix()
    tm.uid = el.get_text('Identifier')
    tm.scale = float(el.get_text('ScaleDenominator'))

    p = u.to_float_pair(el.get_text('TopLeftCorner'))
    tm.x = p[0]
    tm.y = p[1]

    tm.width = int(el.get_text('MatrixWidth'))
    tm.height = int(el.get_text('MatrixHeight'))

    tm.tile_width = int(el.get_text('TileWidth'))
    tm.tile_height = int(el.get_text('TileHeight'))

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
