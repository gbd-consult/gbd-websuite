from PIL import Image
from io import BytesIO

import gws
import gws.gis.layer
import gws.gis.feature
import gws.gis.svg
import gws.tools.misc as misc
import gws.types as t


class Composition:
    def __init__(self, size_px):
        self.image = Image.new('RGBA', size_px, color=0)

    def add_image(self, img: Image.Image, opacity=1):
        img = img.convert('RGBA')

        if img.size != self.image.size:
            gws.log.debug(f'NEEDS_RESIZE self={self.image.size} img={img.size}')
            img = img.resize(size=self.image.size, resample=Image.BILINEAR)

        if opacity < 1:
            alpha = img.getchannel('A').point(lambda x: x * opacity)
            img.putalpha(alpha)

        self.image = Image.alpha_composite(self.image, img)

    def save(self, path):
        self.image.save(path)
        return path


class Renderer:
    def __init__(self):
        self.inp: t.MapRenderInput = None
        self.output: t.MapRenderOutput = None
        self.composition = None

    def run(self, inp: t.MapRenderInput):
        self.inp = inp
        self.output = t.MapRenderOutput({
            'bbox': inp.bbox,
            'dpi': inp.dpi,
            'rotation': inp.rotation,
            'scale': inp.scale,
            'items': []
        })

        # NB: items are top-to-bottom

        for n, item in enumerate(reversed(inp.items)):
            yield item
            self._item(item)

        self._flush_image()

    def _item(self, item: t.MapRenderInputItem):
        try:
            self._try_bitmap(item)
            self._try_bbox(item)
            self._try_svg(item)
            self._try_svg_fragment(item)
            self._try_features(item)
        except Exception:
            # swallow exceptions so that we still can render if some layer fails
            gws.log.error('input item failed')
            gws.log.exception()

    def _try_bitmap(self, item):
        if item.get('bitmap'):
            self._add_image(Image.open(item.bitmap), item.get('opacity') or 1)

    def _try_bbox(self, item):
        if item.get('layer'):
            r = item.layer.render_bbox(
                self.inp.bbox,
                self.inp.map_size_px[0],
                self.inp.map_size_px[1],
                dpi=self.inp.dpi,
                layers=item.sub_layers
            )
            if r:
                opacity = item.get('opacity') or item.layer.opacity or 1
                self._add_image(Image.open(BytesIO(r)), opacity)

    def _try_svg(self, item):
        if item.get('layer'):
            opacity = item.get('opacity') or item.layer.opacity or 1
            # NB: svgs must use document DPI, not the image dpi!
            r = item.layer.render_svg(
                self.inp.bbox,
                misc.PDF_DPI,
                self.inp.scale,
                self.inp.rotation,
                item.get('style'))
            if r:
                self._add_svg(r)

    def _try_svg_fragment(self, item):
        if item.get('svg_fragment'):
            self._add_svg([
                gws.gis.svg.convert_fragment(
                    item.get('svg_fragment'), self.inp.bbox, misc.PDF_DPI, self.inp.scale, self.inp.rotation)
            ])

    def _try_features(self, item):
        fs = item.get('features')
        if fs:
            for f in fs:
                f.set_default_style(item.get('style'))

            # NB: svgs must use document DPI, not the image dpi!
            r = [
                f.to_svg(self.inp.bbox, misc.PDF_DPI, self.inp.scale, self.inp.rotation)
                for f in item.features
            ]
            if r:
                self._add_svg(r)

    def _last_item_is(self, type):
        return self.output.items and self.output.items[-1].type == type

    def _add_image(self, img, opacity):
        if not self._last_item_is('image'):
            self.output.items.append(t.MapRenderOutputItem({
                'type': 'image',
                'image_path': '',
            }))
            self.composition = Composition(self.inp.map_size_px)
        self.composition.add_image(img, opacity)

    def _add_svg(self, svg):
        self._flush_image()
        if not self._last_item_is('svg'):
            self.output.items.append(t.MapRenderOutputItem({
                'type': 'svg',
                'svg_elements': []
            }))

        self.output.items[-1].svg_elements.extend(svg)

    def _flush_image(self):
        if self._last_item_is('image'):
            path = '%s-%s.png' % (self.inp.out_path, len(self.output.items))
            self.composition.save(path)
            self.composition = None
            self.output.items[-1].image_path = path
