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
    metadata: gws.lib.metadata.Record
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str


def parse(xml):
    el = gws.lib.xml2.from_string(xml)

    meta = u.get_meta(el.first('ServiceIdentification'))
    meta['contact'] = u.get_meta_contact(el.first('ServiceProvider.ServiceContact'))
    meta['url'] = u.get_url(el.first('ServiceMetadataURL'))

    source_layers = u.flatten_source_layers(_layer(e) for e in el.all('Contents.Layer'))
    matrix_sets = [_tile_matrix_set(e) for e in el.all('Contents.TileMatrixSet')]

    tms_map = {ms.uid: ms for ms in matrix_sets}

    for sl in source_layers:
        sl.matrix_sets = [tms_map[tid] for tid in sl.matrix_ids]
        sl.supported_crs = sorted(ms.crs for ms in sl.matrix_sets)

    return WMTSCaps(
        matrix_sets=matrix_sets,
        metadata=gws.lib.metadata.Record(meta),
        operations=[gws.OwsOperation(e) for e in u.get_operations(el.first('OperationsMetadata'))],
        source_layers=source_layers,
        supported_crs=sorted(set(ms.crs for ms in matrix_sets)),
        version=el.attr('version'),
    )


def _layer(el):
    oo = gws.lib.gis.SourceLayer()

    oo.metadata = gws.lib.metadata.Record(u.get_meta(el))
    oo.name = oo.metadata.name
    oo.title = oo.metadata.title

    oo.styles = [u.get_style(e) for e in el.all('Style')]
    ds = u.default_style(oo.styles)
    if ds:
        oo.legend_url = ds.legend_url

    oo.supported_bounds = u.get_bounds_list(el)

    oo.is_image = True
    oo.is_visible = True

    oo.matrix_ids = [e.get_text('TileMatrixSet') for e in el.all('TileMatrixSetLink')]
    oo.format = el.get_text('Format')

    oo.resource_urls = {
        rs.attr('resourceType'): rs.attr('template')
        for rs in el.all('ResourceURL')
    }

    return oo


def _tile_matrix_set(el):
    oo = gws.lib.gis.TileMatrixSet()

    oo.uid = el.get_text('Identifier')
    oo.crs = el.get_text('SupportedCRS')
    oo.matrices = sorted(
        [_tile_matrix(e) for e in el.all('TileMatrix')],
        key=lambda m: int('1' + m.uid))

    return oo


def _tile_matrix(el):
    oo = gws.lib.gis.TileMatrix()
    oo.uid = el.get_text('Identifier')
    oo.scale = float(el.get_text('ScaleDenominator'))

    p = u.as_float_pair(el.get_text('TopLeftCorner'))
    oo.x = p[0]
    oo.y = p[1]

    oo.width = int(el.get_text('MatrixWidth'))
    oo.height = int(el.get_text('MatrixHeight'))

    oo.tile_width = int(el.get_text('TileWidth'))
    oo.tile_height = int(el.get_text('TileHeight'))

    oo.extent = _extent_for_matrix(oo)

    return oo


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
