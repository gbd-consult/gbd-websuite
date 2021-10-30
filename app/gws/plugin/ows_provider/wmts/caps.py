import gws
import gws.types as t
import gws.lib.ows.parseutil as u
import gws.lib.net
import gws.lib.gis
import gws.lib.units as units
import gws.lib.xml2
import gws.lib.metadata


# http://portal.opengeospatial.org/files/?artifact_id=35326


class WMTSCaps(gws.Data):
    matrix_sets: t.List[gws.lib.gis.TileMatrixSet]
    metadata: gws.lib.metadata.Metadata
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str


def parse(xml):
    root_el = gws.lib.xml2.from_string(xml)
    source_layers = u.flatten_source_layers(_layer(e) for e in root_el.all('Contents.Layer'))
    matrix_sets = [_tile_matrix_set(e) for e in root_el.all('Contents.TileMatrixSet')]

    tms_map = {ms.uid: ms for ms in matrix_sets}

    for sl in source_layers:
        sl.matrix_sets = [tms_map[tid] for tid in sl.matrix_ids]
        sl.supported_crs = sorted(ms.crs for ms in sl.matrix_sets)

    return WMTSCaps(
        matrix_sets=matrix_sets,
        metadata=u.get_service_metadata(root_el),
        operations=u.get_operations(root_el),
        source_layers=source_layers,
        supported_crs=sorted(set(ms.crs for ms in matrix_sets)),
        version=root_el.attr('version'),
    )


def _layer(el):
    sl = gws.lib.gis.SourceLayer()

    sl.metadata = u.get_metadata(el)
    sl.name = sl.metadata.get('name', '')
    sl.title = sl.metadata.get('title', '')

    sl.styles = [u.get_style(e) for e in el.all('Style')]
    ds = u.default_style(sl.styles)
    if ds:
        sl.legend_url = ds.legend_url

    sl.supported_bounds = u.get_bounds_list(el)

    sl.is_image = True
    sl.is_visible = True

    sl.matrix_ids = [e.get_text('TileMatrixSet') for e in el.all('TileMatrixSetLink')]
    sl.format = el.get_text('Format')

    sl.resource_urls = {
        rs.attr('resourceType'): rs.attr('template')
        for rs in el.all('ResourceURL')
    }

    return sl


def _tile_matrix_set(el):
    tms = gws.lib.gis.TileMatrixSet()

    tms.uid = el.get_text('Identifier')
    tms.crs = el.get_text('SupportedCRS')
    tms.matrices = sorted(
        [_tile_matrix(e) for e in el.all('TileMatrix')],
        key=lambda m: int('1' + m.uid))

    return tms


def _tile_matrix(el):
    tm = gws.lib.gis.TileMatrix()
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

def _extent_for_matrix(m: gws.lib.gis.TileMatrix):
    res = units.scale2res(m.scale)

    return [
        m.x,
        m.y - res * m.height * m.tile_height,
        m.x + res * m.width * m.tile_width,
        m.y,
    ]
