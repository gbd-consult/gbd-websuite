import PIL.Image
import io
import math

import gws
import gws.common.layer
import gws.common.style
import gws.gis.feature
import gws.gis.extent
import gws.tools.svg
import gws.tools.xml2
import gws.gis.renderview

import gws.types as t


#:export
class SvgFragment(t.Data):
    points: t.List[t.Point]
    tags: t.List[t.Tag]
    styles: t.Optional[t.List[t.IStyle]]


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
    view: t.MapRenderView
    items: t.List[t.MapRenderOutputItem]
    base_dir: str


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
            # NB use background for the first composition only
            background = self.ri.background_color
            if any(item.type == 'image' for item in self.output.items):
                background = None
            self.composition = Composition(self.ri.view.size_px, background)
            self.output.items.append(MapRenderOutputItem(type='image'))
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
    vbox = ' '.join(str(s) for s in gws.gis.renderview.pixel_viewbox(ro.view))
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
            tags.append(('svg', gws.tools.svg.SVG_ATTRIBUTES, {'style': css, 'viewBox': vbox}, *item.tags))

    return ''.join(gws.tools.xml2.as_string(tag) for tag in tags)
