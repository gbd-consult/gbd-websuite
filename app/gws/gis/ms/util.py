"""MapServer utility functions."""



import gws
import gws.base.layer
import gws.lib.bounds
import gws.gis.ms

# @TODO check memory usage
MAX_BOX_SIZE = 9000


def raster_render(layer: gws.Layer, lri: gws.LayerRenderInput) -> gws.LayerRenderOutput:
    ts = gws.u.mstime()

    if lri.type == gws.LayerRenderInputType.box:

        def get_box(bounds, width, height):
            ms_map = gws.gis.ms.new_map()
            ms_map.add_layer(layer.msOptions)
            img = ms_map.draw(bounds, (width, height))
            if layer.root.app.developer_option('mapserver.save_temp_maps'):
                gws.u.write_debug_file(f'ms_{layer.uid}_{gws.u.microtime()}.map', ms_map.to_string())
            return img.to_bytes()

        content = gws.base.layer.util.generic_render_box(layer, lri, get_box, box_size=MAX_BOX_SIZE)
        return gws.LayerRenderOutput(content=content)

    if lri.type == gws.LayerRenderInputType.xyz:
        ms_map = gws.gis.ms.new_map()
        ms_map.add_layer(layer.msOptions)

        ext = layer.bounds.extent
        w = (ext[2] - ext[0]) / (1 << lri.z)

        x0 = ext[0] + lri.x * w
        x1 = x0 + w

        y0 = ext[3] - (lri.y + 1) * w
        y1 = y0 + w

        img = ms_map.draw(
            gws.lib.bounds.from_extent((x0, y0, x1, y1), crs=layer.bounds.crs),
            (layer.grid.tileSize, layer.grid.tileSize),
        )
        if layer.root.app.developer_option('mapserver.save_temp_maps'):
            gws.u.write_debug_file(f'ms_{layer.uid}_{gws.u.microtime()}.map', ms_map.to_string())

        if layer.root.app.developer_option('map.annotate_render'):
            ts = gws.u.mstime() - ts
            text = f'{lri.z} : {lri.x} / {lri.y}\nUID={layer.uid}\n{ts}ms'
            img.add_text(text, x=5, y=5).add_box()

        content = img.to_bytes(layer.imageFormat.mimeTypes[0], layer.imageFormat.options)

        return gws.LayerRenderOutput(content=content)
