import gws
import gws.lib.metadata
import gws.base.model
import gws.base.search
import gws.gis.crs
import gws.gis.source
import gws.gis.zoom
import gws.lib.image
import gws.lib.style
import gws.lib.svg
import gws.types as t



def mapproxy_layer_config(layer: gws.ILayer, mc, source_uid):
    mc.layer({
        'name': layer.uid + '_NOCACHE',
        'sources': [source_uid]
    })

    tg = layer.targetGrid

    if tg.corner == 'lt':
        origin = 'nw'
    elif tg.corner == 'lb':
        origin = 'sw'
    else:
        raise gws.Error(f'invalid grid corner {tg.corner!r}')

    tg.uid = mc.grid(gws.compact({
        'origin': origin,
        'tile_size': [tg.tileSize, tg.tileSize],
        'res': tg.resolutions,
        'srs': tg.bounds.crs.epsg,
        'bbox': tg.bounds.extent,
    }))

    front_cache_config = {
        'sources': [source_uid],
        'grids': [tg.uid],
        'cache': {
            'type': 'file',
            'directory_layout': 'mp'
        },
        'meta_size': [1, 1],
        'meta_buffer': 0,
        'disable_storage': True,
        'minimize_meta_requests': True,
        'format': layer.imageFormat,
    }

    cache = getattr(layer, 'cache', None)
    if cache:
        front_cache_config['disable_storage'] = False
        if cache.requestTiles:
            front_cache_config['meta_size'] = [cache.requestTiles, cache.requestTiles]
        if cache.requestBuffer:
            front_cache_config['meta_buffer'] = cache.requestBuffer

    layer.mpxCacheUid = mc.cache(front_cache_config)

    mc.layer({
        'name': layer.uid,
        'sources': [layer.mpxCacheUid]
    })


def mapproxy_back_cache_config(layer: gws.ILayer, mc, url, grid_uid):
    source_uid = mc.source({
        'type': 'tile',
        'url': url,
        'grid': grid_uid,
        'concurrent_requests': layer.var('maxRequests', default=0)
    })

    return mc.cache(gws.compact({
        'sources': [source_uid],
        'grids': [grid_uid],
        'cache': {
            'type': 'file',
            'directory_layout': 'mp'
        },
        'disable_storage': True,
        'format': layer.imageFormat,
    }))


def generic_raster_render(layer: gws.ILayer, lri: gws.LayerRenderInput):
    if lri.type == 'box':
        return gws.LayerRenderOutput(content=generic_render_box_to_bytes(layer, lri))
    if lri.type == 'xyz':
        return gws.LayerRenderOutput(content=generic_render_xyz_to_bytes(layer, lri))


def generic_render_box_to_bytes(layer: gws.ILayer, lri: gws.LayerRenderInput):
    uid = layer.uid
    if not layer.cache:
        uid += '_NOCACHE'

    if not lri.view.rotation:
        return gws.gis.mpx.wms_request(uid, lri.view.bounds, lri.view.size_px[0], lri.view.size_px[1], forward=lri.extraParams)

    # rotation: render a circumsquare around the wanted extent

    circ = gws.gis.extent.circumsquare(lri.view.bounds.extent)
    w, h = lri.view.size_px
    d = gws.gis.extent.diagonal((0, 0, w, h))

    r = gws.gis.mpx.wms_request(
        uid,
        gws.Bounds(crs=lri.view.bounds.crs, extent=circ),
        width=d,
        height=d,
        forward=lri.extraParams)
    if not r:
        return

    img = gws.lib.image.from_bytes(r)

    # rotate the square (NB: PIL rotations are counter-clockwise)
    # and crop the square back to the wanted extent

    img.rotate(-lri.view.rotation).crop((
        d / 2 - w / 2,
        d / 2 - h / 2,
        d / 2 + w / 2,
        d / 2 + h / 2,
    ))

    return img.to_bytes()


def generic_render_xyz_to_bytes(layer: gws.ILayer, lri: gws.LayerRenderInput):
    return gws.gis.mpx.wmts_request(
        layer.uid,
        lri.x,
        lri.y,
        lri.z,
        tile_matrix=layer.targetGrid.uid,
        tile_size=layer.targetGrid.tileSize)
