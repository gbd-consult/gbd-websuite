"""Map render utilities"""

import math

import gws
import gws.types as t
import gws.lib.extent
import gws.lib.feature
import gws.lib.img
import gws.lib.svg
import gws.lib.units as units
import gws.lib.xml2



class Composition:
    def __init__(self, size_px, color=None):
        self.image = gws.lib.img.image_api.new('RGBA', size_px, color)

    def add_image(self, img: gws.lib.img.ImageObject, opacity=1):
        img = img.convert('RGBA')

        if img.size != self.image.size:
            gws.log.debug(f'NEEDS_RESIZE self={self.image.size} img={img.size}')
            img = img.resize(size=self.image.size, resample=gws.lib.img.image_api.BILINEAR)

        if opacity < 1:
            alpha = img.getchannel('A').point(lambda x: x * opacity)
            img.putalpha(alpha)

        self.image = gws.lib.img.image_api.alpha_composite(self.image, img)


class Renderer:
    def run(self, ri: gws.MapRenderInput, base_dir=None):
        self.ri: gws.MapRenderInput = ri
        self.output = gws.MapRenderOutput(
            view=self.ri.view,
            items=[],
            base_dir=base_dir,
        )
        self.default_dpi = self.ri.view.dpi
        self.composition = None

        # NB: items are top-to-bottom

        for item in reversed(self.ri.items):
            yield item
            self._render_item(item)

    def _render_item(self, item: gws.MapRenderInputItem):
        try:
            #  use the item's dpi
            self.ri.view.dpi = item.dpi or self.default_dpi
            self._render_item2(item)
        except Exception:
            # swallow exceptions so that we still can render if some layer fails
            gws.log.error('input item failed')
            gws.log.exception()

    def _render_item2(self, item: gws.MapRenderInputItem):
        # @TODO opacity for svgs

        s = item.opacity
        if s is not None:
            opacity = s
        elif item.layer:
            opacity = item.layer.opacity
        else:
            opacity = 1

        if item.type == gws.MapRenderInputItemType.image:
            self._add_image(item.image, opacity)
            return

        if item.type == gws.MapRenderInputItemType.features:
            for feature in item.features:
                tags = feature.to_svg_tags(self.ri.view, item.style)
                self._add_svg_tags(tags)
            return

        if item.type == gws.MapRenderInputItemType.fragment:
            tags = gws.lib.svg.fragment_tags(item.fragment, self.ri.view)
            self._add_svg_tags(tags)
            return

        if item.type == gws.MapRenderInputItemType.svg_layer:
            tags = item.layer.render_svg_tags(self.ri.view, item.style)
            self._add_svg_tags(tags)
            return

        if item.type == gws.MapRenderInputItemType.image_layer:
            extra_params = {}
            if item.sub_layers:
                extra_params = {'layers': item.sub_layers}
            r = item.layer.render_box(self.ri.view, extra_params)
            if r:
                self._add_image(gws.lib.img.image_from_bytes(r), opacity)

    def _last_item_is(self, type):
        return self.output.items and self.output.items[-1].type == type

    def _add_image(self, img, opacity):
        if not self._last_item_is(gws.MapRenderOutputItemType.image):
            # NB use background for the first composition only
            background = self.ri.background_color
            if any(item.type == gws.MapRenderOutputItemType.image for item in self.output.items):
                background = None
            self.composition = Composition(self.ri.view.size_px, background)
            self.output.items.append(gws.MapRenderOutputItem(type=gws.MapRenderOutputItemType.image))
        self.composition.add_image(img, opacity)
        self.output.items[-1].image = self.composition.image

    def _add_svg_tags(self, tags):
        if not self._last_item_is(gws.MapRenderOutputItemType.svg):
            self.output.items.append(gws.MapRenderOutputItem(type=gws.MapRenderOutputItemType.svg, tags=[]))
        self.output.items[-1].tags.extend(tags)


def output_html(ro: gws.MapRenderOutput) -> str:
    w, h = ro.view.size_mm
    css = ';'.join([
        f'position:absolute',
        f'left:0',
        f'top:0',
        f'width:{w}mm',
        f'height:{h}mm',
    ])
    vbox = ' '.join(str(s) for s in pixel_viewbox(ro.view))
    tags: t.List[gws.Tag] = []

    for item in ro.items:
        if item.type == gws.MapRenderOutputItemType.image:
            path = ro.base_dir + '/' + gws.random_string(64) + '.png'
            t.cast(gws.lib.img.ImageObject, item.image).save(path, 'png')
            tags.append(('img', {'style': css, 'src': path}))
        if item.type == gws.MapRenderOutputItemType.path:
            tags.append(('img', {'style': css, 'src': item.path}))
        if item.type == gws.MapRenderOutputItemType.svg:
            gws.lib.svg.sort_by_z_index(item.tags)
            tags.append(('svg', gws.lib.svg.SVG_ATTRIBUTES, {'style': css, 'viewBox': vbox}, *item.tags))

    return ''.join(gws.lib.xml2.as_string(tag) for tag in tags)


def pixel_transformer(view: gws.MapRenderView):
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

    ox, oy = translate(*gws.lib.extent.center(view.bounds.extent))
    cosa = math.cos(math.radians(view.rotation))
    sina = math.sin(math.radians(view.rotation))

    return fn


def pixel_viewbox(view: gws.MapRenderView):
    """Compute the pixel viewBox for a view"""

    trans = pixel_transformer(view)
    ext = view.bounds.extent
    a = trans(ext[0], ext[1])
    b = trans(ext[2], ext[3])
    return [0, 0, b[0] - a[0], a[1] - b[1]]


def view_from_center(crs: gws.Crs, center: gws.Point, scale: int, out_size: gws.Size, out_size_unit: str, rotation=0, dpi=0):
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
    view.bounds = gws.Bounds(crs=crs, extent=ext)

    return view


def view_from_bbox(crs: gws.Crs, bbox: gws.Extent, out_size: gws.Size, out_size_unit: str, rotation=0, dpi=0):
    """Create a view based on a bounding box"""

    view = _base(out_size, out_size_unit, rotation, dpi)

    view.center = [
        bbox[0] + (bbox[2] - bbox[0]) / 2,
        bbox[1] - (bbox[3] - bbox[1]) / 2,
    ]
    view.scale = units.res2scale((bbox[2] - bbox[0]) / view.size_px[0])
    view.bounds = gws.Bounds(crs=crs, extent=bbox)

    return view


def _base(out_size, out_size_unit, rotation, dpi):
    view = gws.MapRenderView()

    view.dpi = max(units.OGC_SCREEN_PPI, int(dpi))
    view.rotation = rotation

    if out_size_unit == 'px':
        view.size_px = out_size
        view.size_mm = units.point_px2mm(out_size, view.dpi)

    if out_size_unit == 'mm':
        view.size_mm = out_size
        view.size_px = units.point_mm2px(out_size, view.dpi)

    return view
