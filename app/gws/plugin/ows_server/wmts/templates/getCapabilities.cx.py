import gws
import gws.lib.xmlx as xmlx
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
from gws.lib.xmlx import tag


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        tag('Capabilities', doc(ta)),
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

    yield tag(
        'ows:OperationsMetadata',
        tag('ows:Operation', {'name': 'GetCapabilities'}, tpl.ows_service_url(ta)),
        tag('ows:Operation', {'name': 'GetTile'}, tpl.ows_service_url(ta)),
        tag('ows:Operation', {'name': 'GetLegendGraphic'}, tpl.ows_service_url(ta)),
    )

    yield tag('Contents', contents(ta))

    # OGC 07-057r7 Annex D
    if ta.service.metadata.serviceMetadataURL:
        yield tag('ServiceMetadataURL', {'xlink:href': ta.service.metadata.serviceMetadataURL})


def contents(ta: server.TemplateArgs):
    for lc in ta.layerCapsList:
        yield tag('Layer', layer(ta, lc))
    for tms in ta.tileMatrixSets:
        yield tag('TileMatrixSet', tile_matrix_set(ta, tms))


def layer(ta: server.TemplateArgs, lc: server.LayerCaps):
    yield tag('ows:Title', lc.title)

    yield tag('ows:Abstract', lc.layer.metadata.abstract)

    yield tpl.ows_wgs84_bounding_box(lc)

    yield tag('ows:Identifier', lc.layerName)

    yield tag(
        'Style',
        tag('ows:Identifier', 'default'),
        tpl.legend_url(ta, lc) if lc.hasLegend else '',
    )

    yield tag('Format', 'image/png')

    for tms in ta.tileMatrixSets:
        yield tag('TileMatrixSetLink/TileMatrixSet', tms.identifier)


def tile_matrix_set(ta: server.TemplateArgs, tms: gws.TileMatrixSet):
    yield tag('ows:Identifier', tms.identifier)
    yield tag('ows:SupportedCRS', tms.crs.epsg)

    for tm in tms.matrices:
        yield tag(
            'TileMatrix',
            tag('ows:Identifier', tm.identifier),
            tag('ScaleDenominator', tm.scale),
            tag('TopLeftCorner', tm.x, ' ', tm.y),
            tag('TileWidth', tm.tileWidth),
            tag('TileHeight', tm.tileHeight),
            tag('MatrixWidth', tm.width),
            tag('MatrixHeight', tm.height),
        )
