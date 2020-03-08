import PIL.Image
import io
import os

import gws
import gws.common.layer
import gws.gis.feature
import gws.tools.svg
import gws.tools.units as units
import gws.types as t


#:export
class SvgFragment:
    points: t.List[t.Point]
    svg: str


#:export
class RenderView(t.Data):
    bounds: t.Bounds
    center: t.Point
    dpi: int
    rotation: int
    scale: int
    size_mm: t.Size
    size_px: t.Size


#:export
class RenderInputItemType(t.Enum):
    image = 'image'
    features = 'features'
    fragment = 'fragment'
    svg_layer = 'svg_layer'
    image_layer = 'image_layer'


#:export
class RenderInputItem(t.Data):
    type: str = ''
    #:noexport
    image: PIL.Image.Image = None
    features: t.List[t.IFeature]
    layer: t.ILayer = None
    sub_layers: t.List[str] = []
    opacity: float = None
    print_as_vector: bool = None
    style: t.IStyle = None
    fragment: t.SvgFragment = None
    dpi: int = None


#:export
class RenderInput(t.Data):
    view: t.RenderView
    background_color: int
    items: t.List[t.RenderInputItem]


#:export
class RenderOutputItemType(t.Enum):
    image = 'image'
    path = 'path'
    svg = 'svg'


#:export
class RenderOutputItem(t.Data):
    type: str
    #:noexport
    image: PIL.Image.Image = None
    path: str = ''
    elements: t.List[str] = []


#:export
class RenderOutput(t.Data):
    view: 'RenderView'
    items: t.List[RenderOutputItem]


def _view_base(out_size, out_size_unit, rotation, dpi):
    view = t.RenderView()

    view.dpi = max(units.OGC_SCREEN_PPI, int(dpi))
    # @TODO
    view.rotation = 0

    if out_size_unit == 'px':
        view.size_px = out_size
        view.size_mm = units.point_px2mm(out_size, view.dpi)

    if out_size_unit == 'mm':
        view.size_mm = out_size
        view.size_px = units.point_mm2px(out_size, view.dpi)

    return view


def view_from_center(crs: t.Crs, center: t.Point, scale: int, out_size: t.Size, out_size_unit: str, rotation=0, dpi=0):
    view = _view_base(out_size, out_size_unit, rotation, dpi)

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

    view.bounds = t.Bounds(
        crs=crs,
        extent=ext
    )

    return view


def view_from_bbox(crs: t.Crs, bbox: t.Extent, out_size: t.Size, out_size_unit: str, rotation=0, dpi=0):
    view = _view_base(out_size, out_size_unit, rotation, dpi)

    view.center = [
        bbox[0] + (bbox[2] - bbox[0]) / 2,
        bbox[1] - (bbox[3] - bbox[1]) / 2,
    ]
    view.scale = units.res2scale((bbox[2] - bbox[0]) / view.size_px[0])

    view.bounds = t.Bounds(
        crs=crs,
        extent=bbox
    )

    return view


class Composition:
    def __init__(self, size_px, color):
        self.image = PIL.Image.new('RGBA', size_px, color)

    def add_image(self, img: PIL.Image.Image, opacity=1):
        img = img.convert('RGBA')

        if img.size != self.image.size:
            gws.log.debug(f'NEEDS_RESIZE self={self.image.size} img={img.size}')
            img = img.resize(size=self.image.size, resample=PIL.Image.BILINEAR)

        if opacity < 1:
            alpha = img.getchannel('A').point(lambda x: x * opacity)
            img.putalpha(alpha)

        self.image = PIL.Image.alpha_composite(self.image, img)

    def save(self, path):
        self.image.save(path)
        return path


class Renderer:
    def __init__(self):
        self.ri: RenderInput = None
        self.output: RenderOutput = None
        self.composition = None
        self.out_path = ''
        self.default_dpi = 0

    def run(self, ri: RenderInput, out_path=None):
        self.ri = ri
        self.output = RenderOutput(
            view=self.ri.view,
            items=[]
        )
        self.default_dpi = self.ri.view.dpi
        self.out_path = out_path

        # NB: items are top-to-bottom

        for item in reversed(self.ri.items):
            yield item
            self._render_item(item)

        self._flush_image()

    def _render_item(self, item: RenderInputItem):
        try:
            #  use the item's dpi
            self.ri.view.dpi = item.dpi or self.default_dpi
            self._render_item2(item)
        except Exception:
            # swallow exceptions so that we still can render if some layer fails
            gws.log.error('input item failed')
            gws.log.exception()

    def _render_item2(self, item: RenderInputItem):
        # @TODO opacity for svgs

        s = item.opacity
        if s is not None:
            opacity = s
        elif item.layer:
            opacity = item.layer.opacity
        else:
            opacity = 1

        if item.type == t.RenderInputItemType.image:
            self._add_image(item.image, opacity)
            return

        if item.type == t.RenderInputItemType.features:
            r = [
                feature.to_svg(self.ri.view, item.style)
                for feature in item.features
            ]
            self._add_svg(r)
            return

        if item.type == t.RenderInputItemType.fragment:
            svg = gws.tools.svg.convert_fragment(item.fragment, self.ri.view)
            if svg:
                self._add_svg([svg])
            return

        if item.type == t.RenderInputItemType.svg_layer:
            r = item.layer.render_svg(self.ri.view, item.style)
            if r:
                self._add_svg(r)
            return

        if item.type == t.RenderInputItemType.image_layer:
            r = item.layer.render_box(self.ri.view, {'layers': item.sub_layers})
            if r:
                self._add_image(PIL.Image.open(io.BytesIO(r)), opacity)

    def _last_item_is(self, type):
        return self.output.items and self.output.items[-1].type == type

    def _add_image(self, img, opacity):
        if not self._last_item_is(RenderOutputItemType.image):
            self.output.items.append(RenderOutputItem(
                type=RenderOutputItemType.image,
            ))
            self.composition = Composition(self.ri.view.size_px, self.ri.background_color)
        self.composition.add_image(img, opacity)
        self.output.items[-1].image = self.composition.image

    def _add_svg(self, svg):
        self._flush_image()
        if not self._last_item_is(RenderOutputItemType.svg):
            self.output.items.append(RenderOutputItem(
                type=RenderOutputItemType.svg,
                elements=[]
            ))
        self.output.items[-1].elements.extend(svg)

    def _flush_image(self):
        if self._last_item_is(t.RenderOutputItemType.image):
            # path = '%s-%s.png' % (self.out_path, len(self.output.items))
            # self.composition.save(path)
            self.composition = None
            # self.output.items[-1].path = path


def create_html_with_map(html, page_size, margin, out_path, render_output: RenderOutput, map_placeholder):
    map_html = []
    css = 'position: absolute; left: 0; top: 0; width: 100%; height: 100%'
    dir = os.path.dirname(out_path)

    for r in render_output.items:
        if r.type == t.RenderOutputItemType.image:
            path = dir + '/' + gws.random_string(64) + '.png'
            r.image.save(path, 'png')
            map_html.append(f'<img style="{css}" src="{path}"/>')
        if r.type == t.RenderOutputItemType.path:
            map_html.append(f'<img style="{css}" src="{r.path}"/>')
        if r.type == t.RenderOutputItemType.svg:
            s = '\n'.join(r.elements)
            map_html.append(f'<svg style="{css}" version="1.1" xmlns="http://www.w3.org/2000/svg">{s}</svg>')

    return html.replace(map_placeholder, '\n'.join(map_html))
