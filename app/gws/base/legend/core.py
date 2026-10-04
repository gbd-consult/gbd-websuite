"""Base legend object."""

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
    """Base legend object.

    Reads the common ``cacheMaxAge`` and ``options`` settings. Subclasses
    implement ``render`` and can stack several legend images into one with
    ``combine_outputs``.
    """

    cacheMaxAge: int
    """How long legend images from remote sources are cached, in seconds."""
    options: dict
    """Provider-specific legend options."""

    def configure(self):
        self.options = self.cfg('options', default={})
        self.cacheMaxAge = self.cfg('cacheMaxAge', default=3600 * 24)


def combine_outputs(lro_list: list[gws.LegendRenderOutput], options: dict = None) -> Optional[gws.LegendRenderOutput]:
    """Combine several legend outputs into one image.

    The images are stacked vertically, aligned to the left. Outputs without an
    image are skipped.

    Args:
        lro_list: Legend outputs to combine; ``None`` entries are allowed.
        options: Legend options, currently unused.

    Returns:
        A new legend output with the combined image, or ``None`` if there are
        no images.
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
    """Combine images into one."""
    return _combine_vertically(images)


def _combine_vertically(images: list[gws.Image]):
    """Stack images vertically on a transparent canvas."""
    ws = [img.size()[0] for img in images]
    hs = [img.size()[1] for img in images]

    comp = gws.lib.image.from_size((max(ws), sum(hs)))
    y = 0
    for img in images:
        comp.paste(img, (0, y))
        y += img.size()[1]

    return comp
