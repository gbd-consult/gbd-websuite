import gws
import gws.lib.image
import gws.lib.ows.request
import gws.types as t


def render(legend: gws.Legend, context: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    if legend.path:
        return gws.LegendRenderOutput(image_path=legend.path)

    if legend.urls:
        outs = []

        for url in legend.urls:
            try:
                res = gws.lib.ows.request.get_url(url, max_age=legend.cache_max_age)
                if not res.content_type.startswith('image/'):
                    raise gws.lib.ows.error.Error(f'wrong content type {res.content_type!r}')
                img = gws.lib.image.from_bytes(res.content)
                outs.append(gws.LegendRenderOutput(image=img))
            except gws.lib.ows.error.Error:
                gws.log.exception(f'render_legend: download failed url={url!r}')

        # NB even if there's only one image, it's not a bad idea to run it through the image converter
        return _combine_outputs(outs, legend.options)

    if legend.template:
        html = legend.template.render(context or {}).content
        return gws.LegendRenderOutput(html=html)

    if legend.layers:
        return _combine_outputs(
            gws.compact(la.render_legend_with_cache(context) for la in legend.layers if la.has_legend),
            legend.options)

    return None


def to_bytes(out: t.Optional[gws.LegendRenderOutput]) -> t.Optional[bytes]:
    img = to_image(out)
    return img.to_bytes() if img else None


def to_image(out: t.Optional[gws.LegendRenderOutput]) -> t.Optional[gws.IImage]:
    if not out:
        return None
    if out.image:
        return out.image
    if out.image_path:
        return gws.lib.image.from_path(out.image_path)
    if out.html:
        # @TODO render html as image
        pass
    return None


def _combine_outputs(outs: t.List[gws.LegendRenderOutput], options: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    imgs = gws.compact(to_image(out) for out in outs)
    img = _combine_images(imgs, options)
    if not img:
        return None
    return gws.LegendRenderOutput(image=img)


def _combine_images(images: t.List[gws.IImage], options: dict = None) -> t.Optional[gws.IImage]:
    if not images:
        return None
    # @TODO other combination options
    return _combine_vertically(images)


def _combine_vertically(images):
    ws = [img.size[0] for img in images]
    hs = [img.size[1] for img in images]

    out = gws.lib.image.from_size((max(ws), sum(hs)))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.size[1]

    return out
