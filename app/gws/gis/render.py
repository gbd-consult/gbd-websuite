import PIL.Image
import io

import gws
import gws.common.layer
import gws.common.style
import gws.gis.feature
import gws.gis.extent
import gws.tools.svg
import gws.tools.xml2
import gws.tools.units as units

import gws.types as t


#:export
class SvgFragment(t.Data):
    points: t.List[t.Point]
    tags: t.List[t.Tag]
    styles: t.Optional[t.List[t.IStyle]]


#:export
class MapRenderView(t.Data):
    bounds: t.Bounds
    center: t.Point
    dpi: int
    rotation: int
    scale: int
    size_mm: t.Size
    size_px: t.Size


#:export
class MapRenderInputItemType(t.Enum):
    image = 'image'
    features = 'features'
    fragment = 'fragment'
    svg_layer = 'svg_layer'
    image_layer = 'image_layer'


#:export
class MapRenderInputItem(t.Data):
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
class MapRenderInput(t.Data):
    view: t.MapRenderView
    background_color: int
    items: t.List[t.MapRenderInputItem]


#:export
class MapRenderOutputItem(t.Data):
    type: str
    #:noexport
    image: PIL.Image.Image = None
    path: str = ''
    tags: t.List[t.Tag] = []


#:export
class MapRenderOutput(t.Data):
    view: 'MapRenderView'
    items: t.List[MapRenderOutputItem]
    base_dir: str


def _view_base(out_size, out_size_unit, rotation, dpi):
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
    view.bounds = t.Bounds(crs=crs, extent=ext)

    return view


def view_from_bbox(crs: t.Crs, bbox: t.Extent, out_size: t.Size, out_size_unit: str, rotation=0, dpi=0):
    view = _view_base(out_size, out_size_unit, rotation, dpi)

    view.center = [
        bbox[0] + (bbox[2] - bbox[0]) / 2,
        bbox[1] - (bbox[3] - bbox[1]) / 2,
    ]
    view.scale = units.res2scale((bbox[2] - bbox[0]) / view.size_px[0])
    view.bounds = t.Bounds(crs=crs, extent=bbox)

    return view


class Composition:
    def __init__(self, size_px, color=None):
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
    def run(self, ri: MapRenderInput, base_dir=None):
        self.ri: MapRenderInput = ri
        self.output = MapRenderOutput(
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

    def _render_item(self, item: MapRenderInputItem):
        try:
            #  use the item's dpi
            self.ri.view.dpi = item.dpi or self.default_dpi
            self._render_item2(item)
        except Exception:
            # swallow exceptions so that we still can render if some layer fails
            gws.log.error('input item failed')
            gws.log.exception()

    def _render_item2(self, item: MapRenderInputItem):
        # @TODO opacity for svgs

        s = item.opacity
        if s is not None:
            opacity = s
        elif item.layer:
            opacity = item.layer.opacity
        else:
            opacity = 1

        if item.type == t.MapRenderInputItemType.image:
            self._add_image(item.image, opacity)
            return

        if item.type == t.MapRenderInputItemType.features:
            for feature in item.features:
                tags = feature.to_svg_tags(self.ri.view, item.style)
                self._add_svg_tags(tags)
            return

        if item.type == t.MapRenderInputItemType.fragment:
            tags = gws.tools.svg.fragment_tags(item.fragment, self.ri.view)
            self._add_svg_tags(tags)
            return

        if item.type == t.MapRenderInputItemType.svg_layer:
            tags = item.layer.render_svg_tags(self.ri.view, item.style)
            self._add_svg_tags(tags)
            return

        if item.type == t.MapRenderInputItemType.image_layer:
            extra_params = {}
            if item.sub_layers:
                extra_params = {'layers': item.sub_layers}
            r = item.layer.render_box(self.ri.view, extra_params)
            if r:
                self._add_image(PIL.Image.open(io.BytesIO(r)), opacity)

    def _last_item_is(self, type):
        return self.output.items and self.output.items[-1].type == type

    def _add_image(self, img, opacity):
        if not self._last_item_is('image'):
            self.output.items.append(MapRenderOutputItem(type='image'))
            self.composition = Composition(self.ri.view.size_px)
        self.composition.add_image(img, opacity)
        self.output.items[-1].image = self.composition.image

    def _add_svg_tags(self, tags):
        if not self._last_item_is('svg'):
            self.output.items.append(MapRenderOutputItem(type='svg', tags=[]))
        self.output.items[-1].tags.extend(tags)


def output_html(ro: MapRenderOutput) -> str:
    w, h = ro.view.size_mm
    css = ';'.join([
        f'position:absolute',
        f'left:0',
        f'top:0',
        f'width:{w}mm',
        f'height:{h}mm',
    ])
    tags = []

    for item in ro.items:
        if item.type == 'image':
            path = ro.base_dir + '/' + gws.random_string(64) + '.png'
            item.image.save(path, 'png')
            tags.append(('img', {'style': css, 'src': path}))
        if item.type == 'path':
            tags.append(('img', {'style': css, 'src': item.path}))
        if item.type == 'svg':
            gws.tools.svg.sort_by_z_index(item.tags)
            tags.append(('svg', gws.tools.svg.SVG_ATTRIBUTES, {'style': css}, *item.tags))

    return ''.join(gws.tools.xml2.as_string(tag) for tag in tags)
