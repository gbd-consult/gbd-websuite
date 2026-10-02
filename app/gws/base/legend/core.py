from typing import Optional

import gws
import gws.lib.image


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    """Layer legend confuguration."""

    cacheMaxAge: gws.Duration = '1d'
    """How long legend images from remote sources are cached."""
    options: Optional[dict]
    """Provider-specific legend options."""


class Object(gws.Legend):
    """Generic legend object."""

    cacheMaxAge: int
    options: dict

    def configure(self):
        self.options = self.cfg('options', default={})
        self.cacheMaxAge = self.cfg('cacheMaxAge', default=3600 * 24)


def combine_outputs(lro_list: list[gws.LegendRenderOutput], options: dict = None) -> Optional[gws.LegendRenderOutput]:
    """Combine multiple LegendRenderOutputs into a single output.

    Args:
        lro_list: A list of legend render outputs to combine.
        options: Optional combination settings (currently unused).

    Returns:
        A new LegendRenderOutput containing the combined image,
        or None if no images were provided.
    """
    imgs = []
    for lro in lro_list:
        if lro and lro.image:
            imgs.append(lro.image)
    if not imgs:
        return
    img = _combine_images(imgs, options)
    return gws.LegendRenderOutput(image=img, size=img.size())


def _combine_images(images: list[gws.Image], options: dict = None):
    return _combine_vertically(images)


def _combine_vertically(images: list[gws.Image]):
    ws = [img.size()[0] for img in images]
    hs = [img.size()[1] for img in images]

    comp = gws.lib.image.from_size((max(ws), sum(hs)))
    y = 0
    for img in images:
        comp.paste(img, (0, y))
        y += img.size()[1]

    return comp
