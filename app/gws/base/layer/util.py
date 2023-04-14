import math

import gws
import gws.base.model
import gws.base.search
import gws.gis.crs
import gws.gis.extent
import gws.gis.mpx
import gws.gis.source
import gws.gis.zoom
import gws.lib.image
import gws.lib.metadata
import gws.lib.style
import gws.lib.svg

import gws.types as t


def mapproxy_layer_config(layer: gws.ILayer, mc, source_uid):
    mc.layer({
        'name': layer.uid + '_NOCACHE',
        'sources': [source_uid]
    })

    tg = layer.grid

    tg.uid = mc.grid(gws.compact({
        'origin': tg.corner,
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
        'concurrent_requests': layer.cfg('maxRequests', default=0)
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


##

_BOX_SIZE = 1000
_BOX_BUFFER = 200

_GetBoxFn = t.Callable[[gws.Bounds, float, float], bytes]


def mpx_raster_render(layer: gws.ILayer, lri: gws.LayerRenderInput):
    if lri.type == 'box':

        uid = layer.uid
        if not layer.cache:
            uid += '_NOCACHE'

        def get_box(bounds, width, height):
            return gws.gis.mpx.wms_request(uid, bounds, width, height, forward=lri.extraParams)

        content = generic_render_box(layer, lri, get_box)
        return gws.LayerRenderOutput(content=content)

    if lri.type == 'xyz':
        content = gws.gis.mpx.wmts_request(
            layer.uid,
            lri.x,
            lri.y,
            lri.z,
            tile_matrix=layer.grid.uid,
            tile_size=layer.grid.tileSize)

        annotate = layer.root.app.developer_option('map.annotate_render')
        if annotate:
            content = _annotate(content, f'{lri.x} {lri.y} {lri.z}')

        return gws.LayerRenderOutput(content=content)


def generic_render_box(layer: gws.ILayer, lri: gws.LayerRenderInput, get_box: _GetBoxFn) -> bytes:
    annotate = layer.root.app.developer_option('map.annotate_render')

    max_box_size = lri.boxSize or _BOX_SIZE
    box_buffer = lri.boxBuffer or _BOX_BUFFER

    w, h = lri.view.pxSize

    if not lri.view.rotation and w < max_box_size and h < max_box_size:
        # fast path: no rotation, small box
        content = get_box(lri.view.bounds, w, h)
        if annotate:
            content = _annotate(content, 'fast')
        return content

    if not lri.view.rotation:
        # no rotation, big box
        img = _box_to_image(lri.view.bounds, w, h, max_box_size, box_buffer, annotate, get_box)
        return img.to_bytes()

    # rotation: render a circumsquare around the wanted extent

    circ = gws.gis.extent.circumsquare(lri.view.bounds.extent)
    d = gws.gis.extent.diagonal((0, 0, w, h))
    b = gws.Bounds(crs=lri.view.bounds.crs, extent=circ)

    img = _box_to_image(b, d, d, max_box_size, box_buffer, annotate, get_box)

    # rotate the square (NB: PIL rotations are counter-clockwise)
    # and crop the square back to the wanted extent

    img.rotate(-lri.view.rotation).crop((
        d / 2 - w / 2,
        d / 2 - h / 2,
        d / 2 + w / 2,
        d / 2 + h / 2,
    ))

    return img.to_bytes()


def _box_to_image(bounds: gws.Bounds, width: float, height: float, max_size: int, buf: int, annotate: bool, get_box: _GetBoxFn) -> gws.lib.image.Image:

    if width < max_size and height < max_size:
        content = get_box(bounds, width, height)
        img = gws.lib.image.from_bytes(content)
        if annotate:
            img = _annotate_image(img, 'small')
        return img

    xcount = math.ceil(width / max_size)
    ycount = math.ceil(height / max_size)

    ext = bounds.extent

    xres = (ext[2] - ext[0]) / width
    yres = (ext[3] - ext[1]) / height

    ext_w = xres * max_size
    ext_h = yres * max_size

    grid = []

    for ny in range(ycount):
        for nx in range(xcount):
            e = (
                ext[0] + ext_w * (nx + 0) - buf * xres,
                ext[3] - ext_h * (ny + 1) - buf * yres,
                ext[0] + ext_w * (nx + 1) + buf * xres,
                ext[3] - ext_h * (ny + 0) + buf * yres,
            )
            bounds = gws.Bounds(crs=bounds.crs, extent=e)
            content = get_box(bounds, max_size + buf * 2, max_size + buf * 2)
            grid.append([nx, ny, content])

    img = gws.lib.image.from_size((max_size * xcount, max_size * ycount))

    for nx, ny, content in grid:
        tile = gws.lib.image.from_bytes(content)
        tile.crop((buf, buf, tile.size()[0] - buf, tile.size()[1] - buf))
        if annotate:
            _annotate_image(tile, f'{nx} {ny}')
        img.paste(tile, (nx * max_size, ny * max_size))

    img.crop((0, 0, gws.to_rounded_int(width), gws.to_rounded_int(height)))
    return img


def _annotate(content, text):
    return _annotate_image(gws.lib.image.from_bytes(content), text).to_bytes()


def _annotate_image(img, text):
    return img.add_text(text, x=5, y=5).add_box()
