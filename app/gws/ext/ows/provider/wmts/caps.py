import gws
import gws.common.ows.provider.parseutil as u
import gws.tools.net
import gws.tools.xml3
import gws.tools.units as units
import gws.types as t

from . import types


# http://portal.opengeospatial.org/files/?artifact_id=35326


def parse(prov, xml):
    el = gws.tools.xml3.from_string(xml)

    prov.meta = t.MetaData(u.get_meta(el.first('ServiceIdentification')))
    prov.meta.contact = t.MetaContact(u.get_meta_contact(el.first('ServiceProvider.ServiceContact')))

    if not prov.meta.url:
        prov.meta.url = u.get_url(el.first('ServiceMetadataURL'))

    prov.operations = u.get_operations(el.first('OperationsMetadata'))
    prov.version = el.attr('version')

    prov.source_layers = u.flatten_source_layers(_layer(e) for e in el.all('Contents.Layer'))

    tms_all = {}
    for e in el.all('Contents.TileMatrixSet'):
        tms = _tile_matrix_set(e)
        tms_all[tms.uid] = tms

    for sl in prov.source_layers:
        sl.matrix_sets = []
        cs = set()

        for tid in sl.matrix_ids:
            tms = tms_all[tid]
            cs.add(tms.crs)
            sl.matrix_sets.append(tms)

        sl.supported_crs = sorted(cs)

    prov.supported_crs = u.crs_from_layers(prov.source_layers)


def _layer(el):
    oo = types.SourceLayer()

    oo.meta = t.MetaData(u.get_meta(el))
    oo.name = oo.meta.name
    oo.title = oo.meta.title

    oo.styles = [u.get_style(e) for e in el.all('Style')]
    ds = u.default_style(oo.styles)
    if ds:
        oo.legend = ds.legend

    oo.extents = u.get_extents(el)

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
    oo = types.TileMatrixSet()

    oo.uid = el.get_text('Identifier')
    oo.crs = el.get_text('SupportedCRS')
    oo.matrices = sorted(
        [_tile_matrix(e) for e in el.all('TileMatrix')],
        key=lambda m: int('1' + m.uid))

    return oo


def _tile_matrix(el):
    oo = types.TileMatrix()
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

def _extent_for_matrix(m: types.TileMatrix):
    res = units.scale2res(m.scale)

    return [
        m.x,
        m.y - res * m.height * m.tile_height,
        m.x + res * m.width * m.tile_width,
        m.y,
    ]
