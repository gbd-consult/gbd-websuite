from io import BytesIO

import gws
import gws.types as t
import gws.lib.ows.request
import gws.lib.img


def render(legend: gws.Legend, context: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    if legend.path:
        return gws.LegendRenderOutput(image_path=legend.path)

    if legend.urls:
        buf = []

        for url in legend.urls:
            try:
                res = gws.lib.ows.request.get_url(url, max_age=legend.cache_max_age)
                if not res.content_type.startswith('image/'):
                    raise gws.lib.ows.error.Error(f'wrong content type {res.content_type!r}')
                buf.append(res.content)
            except gws.lib.ows.error.Error as exc:
                gws.log.exception(f'render_legend: download failed url={url!r}')

        # NB even if there's only one image, it's not a bad idea to run it through the image converter
        return gws.LegendRenderOutput(image=_combine_images(buf, legend.options))

    if legend.template:
        html = legend.template.render(context or {}).content
        return gws.LegendRenderOutput(html=html)

    return None


def as_bytes(out: t.Optional[gws.LegendRenderOutput]) -> t.Optional[bytes]:
    if not out:
        return None
    if out.image:
        return out.image
    if out.image_path:
        return gws.read_file_b(out.image_path)
    if out.html:
        # @TODO render html as image
        pass
    return None


def combine_urls(urls: t.List[str], options: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    outs = gws.compact(render(gws.Legend(url=url)) for url in urls)
    return combine_outputs(outs, options)


def combine_outputs(outs: t.List[gws.LegendRenderOutput], options: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    buf = gws.filter(as_bytes(out) for out in outs)
    img = _combine_images(buf, options)
    if not img:
        return None
    return gws.LegendRenderOutput(image=img)


def _combine_images(buf: t.List[bytes], options: dict = None) -> t.Optional[bytes]:
    images = [gws.lib.img.image_from_bytes(b) for b in buf if b]
    if not images:
        return None
    # @TODO other combination options
    return _combine_vertically(images)


def _combine_vertically(images):
    ws = [img.size[0] for img in images]
    hs = [img.size[1] for img in images]

    out = gws.lib.img.image_api.new('RGBA', (max(ws), sum(hs)), (0, 0, 0, 0))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.size[1]

    return gws.lib.img.image_to_bytes(out)
