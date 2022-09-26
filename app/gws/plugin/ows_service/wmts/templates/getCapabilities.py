import gws.lib.xmlx as xmlx
import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def layer(lc):
        yield 'ows:Title', lc.title

        if lc.meta.abstract:
            yield 'ows:Abstract', lc.meta.abstract

        yield tpl.ows_wgs84_bounding_box(lc)

        yield 'ows:Identifier', lc.layer_pname

        if lc.has_legend:
            yield (
                'Style',
                ('ows:Identifier', 'default'),
                ('ows:Title', 'default'),
                tpl.legendUrl(ARGS, lc))

        yield 'Format', 'image/png'

        for tms in ARGS.tileMatrixSets:
            yield 'TileMatrixSetLink TileMatrixSet', tms.uid

    def matrix_set(tms):
        yield 'ows:Identifier', tms.uid
        yield 'ows:SupportedCRS', tms.crs.epsg

        for tm in tms.matrices:
            yield (
                'TileMatrix',
                ('ows:Identifier', tm.uid),
                ('ScaleDenominator', tm.scale),
                ('TopLeftCorner', tm.x, ' ', tm.y),
                ('TileWidth', tm.tile_width),
                ('TileHeight', tm.tile_height),
                ('MatrixWidth', tm.width),
                ('MatrixHeight', tm.height),
            )

    def contents():
        for lc in ARGS.layer_caps_list:
            yield 'Layer', layer(lc)
        for tms in ARGS.tileMatrixSets:
            yield 'TileMatrixSet', matrix_set(tms)

    def doc():
        yield {'xmlns': 'wmts', 'version': ARGS.version}

        yield tpl.ows_service_identification(ARGS)
        yield tpl.ows_service_provider(ARGS)

        yield (
            'ows:OperationsMetadata',
            ('ows:Operation', {'name': 'GetCapabilities'}, tpl.ows_service_url(ARGS)),
            ('ows:Operation', {'name': 'GetTile'}, tpl.ows_service_url(ARGS)))

        yield 'Contents', contents()

    return tpl.to_xml(ARGS, ('Capabilities', doc()))
