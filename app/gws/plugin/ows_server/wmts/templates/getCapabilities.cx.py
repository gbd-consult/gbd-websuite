import gws
import gws.lib.xmlx as xmlx
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
from gws.lib.xmlx import tag


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        tag('WMTS:Capabilities', doc(ta)),
        default_namespace=xmlx.namespace.c.WMTS,
    )


def doc(ta: server.TemplateArgs):
    yield {
        'version': ta.version,
        'updateSequence': ta.service.updateSequence,
    }

    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)

    yield tag(
        'OWS_11:OperationsMetadata',
        tag('OWS_11:Operation', {'name': 'GetCapabilities'}, tpl.ows_service_url(ta)),
        tag('OWS_11:Operation', {'name': 'GetTile'}, tpl.ows_service_url(ta)),
        tag('OWS_11:Operation', {'name': 'GetLegendGraphic'}, tpl.ows_service_url(ta)),
    )

    yield tag('WMTS:Contents', contents(ta))

    # OGC 07-057r7 Annex D
    if ta.service.metadata.serviceMetadataURL:
        yield tag('WMTS:ServiceMetadataURL', {'XLINK:href': ta.service.metadata.serviceMetadataURL})


def contents(ta: server.TemplateArgs):
    for lc in ta.layerCapsList:
        yield tag('WMTS:Layer', layer(ta, lc))
    for tms in ta.tileMatrixSets:
        yield tag('WMTS:TileMatrixSet', tile_matrix_set(ta, tms))


def layer(ta: server.TemplateArgs, lc: server.LayerCaps):
    yield tag('OWS_11:Title', lc.title)
    yield tag('OWS_11:Abstract', lc.layer.metadata.abstract)
    yield tpl.ows_wgs84_bounding_box(lc)
    yield tag('OWS_11:Identifier', lc.layerName)

    yield tag(
        'Style',
        tag('OWS_11:Identifier', 'default'),
        tpl.legend_url(ta, lc) if lc.hasLegend else '',
    )

    yield tag('WMTS:Format', 'image/png')

    for tms in ta.tileMatrixSets:
        yield tag('WMTS:TileMatrixSetLink/WMTS:TileMatrixSet', tms.identifier)


def tile_matrix_set(ta: server.TemplateArgs, tms: gws.TileMatrixSet):
    yield tag('OWS_11:Identifier', tms.identifier)
    yield tag('OWS_11:SupportedCRS', tms.crs.epsg)

    for tm in tms.matrices:
        yield tag(
            'TileMatrix',
            tag('OWS_11:Identifier', tm.identifier),
            tag('WMTS:ScaleDenominator', tm.scale),
            tag('WMTS:TopLeftCorner', tm.x, ' ', tm.y),
            tag('WMTS:TileWidth', tm.tileWidth),
            tag('WMTS:TileHeight', tm.tileHeight),
            tag('WMTS:MatrixWidth', tm.width),
            tag('WMTS:MatrixHeight', tm.height),
        )
