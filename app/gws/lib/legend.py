import gws
import gws.lib.image
import gws.lib.html2
import gws.lib.mime
import gws.lib.units as units
import gws.lib.ows.request
import gws.types as t


def render(legend: gws.Legend, context: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    if legend.path:
        img = gws.lib.image.from_path(legend.path)
        return gws.LegendRenderOutput(image=img, size=img.size)

    if legend.urls:
        lros = []

        for url in legend.urls:
            try:
                res = gws.lib.ows.request.get_url(url, max_age=legend.cache_max_age)
                if not res.content_type.startswith('image/'):
                    raise gws.lib.ows.error.Error(f'wrong content type {res.content_type!r}')
                img = gws.lib.image.from_bytes(res.content)
                lro = gws.LegendRenderOutput(image=img, size=img.size)
                lros.append(lro)
            except gws.lib.ows.error.Error:
                gws.log.exception(f'render_legend: download failed url={url!r}')

        # NB even if there's only one image, it's not a bad idea to run it through the image converter
        return _combine_outputs(gws.compact(lros), legend.options)

    if legend.template:
        # @TODO return html legends as html
        out_path = gws.TMP_DIR + '/' + gws.random_string(64) + '.png'
        tro = legend.template.render(gws.TemplateRenderInput(
            context=context,
            out_path=out_path,
            out_mime=gws.lib.mime.PNG))
        img = gws.lib.image.from_path(tro.path)
        return gws.LegendRenderOutput(image=img, size=img.size)

    if legend.layers:
        lros = gws.compact(la.render_legend_with_cache(context) for la in legend.layers if la.has_legend)
        return _combine_outputs(lros, legend.options)

    return None


def to_bytes(lro: gws.LegendRenderOutput) -> t.Optional[bytes]:
    img = to_image(lro)
    return img.to_bytes() if img else None


def to_image(lro: gws.LegendRenderOutput) -> t.Optional[gws.IImage]:
    if lro.image:
        return lro.image
    if lro.image_path:
        return gws.lib.image.from_path(lro.image_path)
    if lro.html:
        return None


def to_image_path(lro: gws.LegendRenderOutput, out_path: str) -> t.Optional[str]:
    if lro.image:
        return lro.image.to_path(out_path + '.png', gws.lib.mime.PNG)
    if lro.image_path:
        return lro.image_path
    if lro.html:
        return None


def _combine_outputs(lros: t.List[gws.LegendRenderOutput], options: dict = None) -> t.Optional[gws.LegendRenderOutput]:
    imgs = gws.compact(to_image(lro) for lro in lros)
    img = _combine_images(imgs, options)
    if not img:
        return None
    return gws.LegendRenderOutput(image=img, size=img.size)


def _combine_images(images: t.List[gws.IImage], options: dict = None) -> t.Optional[gws.IImage]:
    if not images:
        return None
    # @TODO other combination options
    return _combine_vertically(images)


def _combine_vertically(images):
    ws = [img.size[0] for img in images]
    hs = [img.size[1] for img in images]

    comp = gws.lib.image.from_size((max(ws), sum(hs)))
    y = 0
    for img in images:
        comp.paste(img, (0, y))
        y += img.size[1]

    return comp
