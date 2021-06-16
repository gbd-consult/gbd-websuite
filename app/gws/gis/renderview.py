import math

import gws
import gws.gis.extent
import gws.lib.units as units

import gws.types as t


#:export
class MapRenderView(t.Data):
    bounds: t.Bounds
    center: t.Point
    dpi: int
    rotation: int
    scale: int
    size_mm: t.Size
    size_px: t.Size


def pixel_transformer(view: t.MapRenderView):
    """Create a pixel transformer f(map_x, map_y) -> (pixel_x, pixel_y) for a view"""

    # @TODO cache the transformer

    def translate(x, y):
        x = x - view.bounds.extent[0]
        y = view.bounds.extent[3] - y

        return (
            units.mm2px_f((x / view.scale) * 1000, view.dpi),
            units.mm2px_f((y / view.scale) * 1000, view.dpi))

    def rotate(x, y):
        return (
            cosa * (x - ox) - sina * (y - oy) + ox,
            sina * (x - ox) + cosa * (y - oy) + oy)

    def fn(x, y):
        x, y = translate(x, y)
        if view.rotation:
            x, y = rotate(x, y)
        return x, y

    ox, oy = translate(*gws.gis.extent.center(view.bounds.extent))
    cosa = math.cos(math.radians(view.rotation))
    sina = math.sin(math.radians(view.rotation))

    return fn


def pixel_viewbox(view: t.MapRenderView):
    """Compute the pixel viewBox for a view"""

    trans = pixel_transformer(view)
    ext = view.bounds.extent
    a = trans(ext[0], ext[1])
    b = trans(ext[2], ext[3])
    return [0, 0, b[0] - a[0], a[1] - b[1]]


def from_center(crs: t.Crs, center: t.Point, scale: int, out_size: t.Size, out_size_unit: str, rotation=0, dpi=0):
    """Create a view based on a center point"""

    view = _base(out_size, out_size_unit, rotation, dpi)

    view.center = center
    view.scale = scale

    # @TODO assuming projection units are 'm'
    unit_per_mm = scale / 1000.0

    w = units.px2mm(view.size_px[0], view.dpi)
    h = units.px2mm(view.size_px[1], view.dpi)

    ext = [
        view.center[0] - (w * unit_per_mm) / 2,
        view.center[1] - (h * unit_per_mm) / 2,
        view.center[0] + (w * unit_per_mm) / 2,
        view.center[1] + (h * unit_per_mm) / 2,
    ]
    view.bounds = t.Bounds(crs=crs, extent=ext)

    return view


def from_bbox(crs: t.Crs, bbox: t.Extent, out_size: t.Size, out_size_unit: str, rotation=0, dpi=0):
    """Create a view based on a bounding box"""

    view = _base(out_size, out_size_unit, rotation, dpi)

    view.center = [
        bbox[0] + (bbox[2] - bbox[0]) / 2,
        bbox[1] - (bbox[3] - bbox[1]) / 2,
    ]
    view.scale = units.res2scale((bbox[2] - bbox[0]) / view.size_px[0])
    view.bounds = t.Bounds(crs=crs, extent=bbox)

    return view


def _base(out_size, out_size_unit, rotation, dpi):
    view = t.MapRenderView()

    view.dpi = max(units.OGC_SCREEN_PPI, int(dpi))
    view.rotation = rotation

    if out_size_unit == 'px':
        view.size_px = out_size
        view.size_mm = units.point_px2mm(out_size, view.dpi)

    if out_size_unit == 'mm':
        view.size_mm = out_size
        view.size_px = units.point_mm2px(out_size, view.dpi)

    return view
