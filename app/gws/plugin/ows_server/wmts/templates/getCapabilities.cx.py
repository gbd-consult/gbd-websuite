import gws
import gws.lib.xmlx as xmlx
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        ('Capabilities', doc(ta)),
        namespaces={
            'ows': xmlx.namespace.require('ows11'),
        },
        default_namespace=xmlx.namespace.require('wmts'),
    )


def doc(ta: server.TemplateArgs):
    yield {
        'version': ta.version,
        'updateSequence': ta.service.updateSequence,
    }

    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)

    yield (
        'ows:OperationsMetadata',
        ('ows:Operation', {'name': 'GetCapabilities'}, tpl.ows_service_url(ta)),
        ('ows:Operation', {'name': 'GetTile'}, tpl.ows_service_url(ta)),
        ('ows:Operation', {'name': 'GetLegendGraphic'}, tpl.ows_service_url(ta)),
    )

    yield 'Contents', contents(ta)

    # OGC 07-057r7 Annex D
    if ta.service.metadata.serviceMetadataURL:
        yield 'ServiceMetadataURL', {'xlink:href': ta.service.metadata.serviceMetadataURL}


def contents(ta: server.TemplateArgs):
    for lc in ta.layerCapsList:
        yield 'Layer', layer(ta, lc)
    for tms in ta.tileMatrixSets:
        yield 'TileMatrixSet', tile_matrix_set(ta, tms)


def layer(ta: server.TemplateArgs, lc: server.LayerCaps):
    yield 'ows:Title', lc.title

    yield 'ows:Abstract', lc.layer.metadata.abstract

    yield tpl.ows_wgs84_bounding_box(lc)

    yield 'ows:Identifier', lc.layerName

    if lc.hasLegend:
        yield (
            'Style',
            ('ows:Identifier', 'default'),
            tpl.legend_url(ta, lc),
        )

    yield 'Format', 'image/png'

    for tms in ta.tileMatrixSets:
        yield 'TileMatrixSetLink/TileMatrixSet', tms.uid


def tile_matrix_set(ta: server.TemplateArgs, tms: gws.TileMatrixSet):
    yield 'ows:Identifier', tms.uid
    yield 'ows:SupportedCRS', tms.crs.epsg

    for tm in tms.matrices:
        yield (
            'TileMatrix',
            ('ows:Identifier', tm.uid),
            ('ScaleDenominator', tm.scale),
            ('TopLeftCorner', tm.x, ' ', tm.y),
            ('TileWidth', tm.tileWidth),
            ('TileHeight', tm.tileHeight),
            ('MatrixWidth', tm.width),
            ('MatrixHeight', tm.height),
        )
