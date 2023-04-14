import gws
import gws.gis.extent
import gws.lib.image
import gws.lib.mime
import gws.types as t

from . import provider


def box_to_bytes(layer: gws.ILayer, view: gws.MapView, params: dict) -> bytes:
    # boxes larger than that will be tiled in _box_request
    size_threshold = 2500

    if not view.rotation:
        return _box_request(layer, view.bounds, view.pxSize[0], view.pxSize[1], params, tile_size=size_threshold)

    # @TODO merge with layer/util/generic_render

    circ = gws.gis.extent.circumsquare(view.bounds.extent)
    w, h = view.pxSize
    d = gws.gis.extent.diagonal((0, 0, w, h))

    r = _box_request(layer, gws.Bounds(crs=view.bounds.crs, extent=circ), d, d, params, tile_size=size_threshold)

    img = gws.lib.image.from_bytes(r)

    img.rotate(-view.rotation).crop((
        d / 2 - w / 2,
        d / 2 - h / 2,
        d / 2 + w / 2,
        d / 2 + h / 2,
    ))

    return img.to_bytes()


def _box_request(layer: gws.ILayer, bounds, width, height, params, tile_size):
    if width < tile_size and height < tile_size:
        return t.cast(provider.Object, layer.provider).get_map(layer, bounds, width, height, params)

    # xcount = math.ceil(width / tile_size)
    # ycount = math.ceil(height / tile_size)
    #
    # ext = bounds.extent
    #
    # bw = (ext[2] - ext[0]) * tile_size / width
    # bh = (ext[3] - ext[1]) * tile_size / height
    #
    # grid = []
    #
    # for ny in range(ycount):
    #     for nx in range(xcount):
    #         e = [
    #             ext[0] + bw * nx,
    #             ext[3] - bh * (ny + 1),
    #             ext[0] + bw * (nx + 1),
    #             ext[3] - bh * ny,
    #         ]
    #         bounds = t.Bounds(crs=bounds.crs, extent=e)
    #         content = _get_map_request(layer, bounds, tile_size, tile_size, params)
    #         grid.append([nx, ny, content])
    #
    # out = PIL.Image.new('RGBA', (tile_size * xcount, tile_size * ycount), (0, 0, 0, 0))
    # for nx, ny, content in grid:
    #     img = PIL.Image.open(io.BytesIO(content))
    #     out.paste(img, (nx * tile_size, ny * tile_size))
    #
    # out = out.crop((0, 0, width, height))
    #
    # buf = io.BytesIO()
    # out.save(buf, 'PNG')
    # return buf.getvalue()



    return int(round((x * ppi) / MM_PER_IN))


