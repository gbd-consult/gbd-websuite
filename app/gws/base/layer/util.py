from typing import Callable

import math

import gws
import gws.base.model
import gws.base.search
import gws.lib.crs
import gws.lib.extent
import gws.gis.source
import gws.gis.zoom
import gws.lib.image
import gws.base.metadata
import gws.lib.style
import gws.lib.svg



##

_DEFAULT_BOX_SIZE = 1000
_DEFAULT_BOX_BUFFER = 200

_GetBoxFn = Callable[[gws.Bounds, float, float], bytes]


def generic_render_box(layer: gws.Layer, lri: gws.LayerRenderInput, get_box: _GetBoxFn, box_size: int = 0, box_buffer: int = 0) -> bytes:
    annotate = layer.root.app.developer_option('map.annotate_render')

    box_size = box_size or _DEFAULT_BOX_SIZE
    box_buffer = box_buffer or _DEFAULT_BOX_BUFFER

    w, h = lri.view.pxSize

    if not lri.view.rotation and w < box_size and h < box_size:
        # fast path: no rotation, small box
        content = get_box(lri.view.bounds, w, h)
        if annotate:
            content = _annotate(content, 'fast')
        return content

    if not lri.view.rotation:
        # no rotation, big box
        img = _box_to_image(lri.view.bounds, w, h, box_size, box_buffer, annotate, get_box)
        return img.to_bytes()

    # rotation: render a circumsquare around the wanted extent

    circ = gws.lib.extent.circumsquare(lri.view.bounds.extent)
    d = gws.lib.extent.diagonal((0, 0, w, h))
    b = gws.Bounds(crs=lri.view.bounds.crs, extent=circ)

    img = _box_to_image(b, d, d, box_size, box_buffer, annotate, get_box)

    # rotate the square (NB: PIL rotations are counter-clockwise)
    # and crop the square back to the wanted extent

    img.rotate(-lri.view.rotation).crop((
        d / 2 - w / 2,
        d / 2 - h / 2,
        d / 2 + w / 2,
        d / 2 + h / 2,
    ))

    return img.to_bytes()


def _box_to_image(bounds: gws.Bounds, width: float, height: float, max_size: int, buffer: int, annotate: bool, get_box: _GetBoxFn) -> gws.lib.image.Image:

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

    gws.log.debug(f'_box_to_image (BIG): {xcount=} {ycount=} {xres=} {yres=}')

    ext_w = xres * max_size
    ext_h = yres * max_size

    grid = []

    for ny in range(ycount):
        for nx in range(xcount):
            e = (
                ext[0] + ext_w * (nx + 0) - buffer * xres,
                ext[3] - ext_h * (ny + 1) - buffer * yres,
                ext[0] + ext_w * (nx + 1) + buffer * xres,
                ext[3] - ext_h * (ny + 0) + buffer * yres,
            )
            bounds = gws.Bounds(crs=bounds.crs, extent=e)
            content = get_box(bounds, max_size + buffer * 2, max_size + buffer * 2)
            gws.log.debug(f'_box_to_image (BIG): {nx=}/{xcount} {ny=}/{ycount} {len(content)=}')
            grid.append([nx, ny, content])

    img = gws.lib.image.from_size((max_size * xcount, max_size * ycount))

    for nx, ny, content in grid:
        tile = gws.lib.image.from_bytes(content)
        tile.crop((buffer, buffer, tile.size()[0] - buffer, tile.size()[1] - buffer))
        if annotate:
            _annotate_image(tile, f'{nx} {ny}')
        img.paste(tile, (nx * max_size, ny * max_size))

    img.crop((0, 0, gws.u.to_rounded_int(width), gws.u.to_rounded_int(height)))
    return img


def _annotate(content, text):
    return _annotate_image(gws.lib.image.from_bytes(content), text).to_bytes()


def _annotate_image(img, text):
    return img.add_text(text, x=5, y=5).add_box()
