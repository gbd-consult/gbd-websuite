from typing import Optional

import gws
import gws.lib.image
import gws.lib.mime


class Props(gws.Props):
    type: str


class Config(gws.ConfigWithAccess):
    """Layer legend confuguration."""

    cacheMaxAge: gws.Duration = '1d'
    """Max cache age for remote legends."""
    options: Optional[dict]
    """Provider-dependent legend options."""


class Object(gws.Legend):
    """Generic legend object."""

    cacheMaxAge: int
    options: dict

    def configure(self):
        self.options = self.cfg('options', default={})
        self.cacheMaxAge = self.cfg('cacheMaxAge', default=3600 * 24)


def output_to_bytes(lro: gws.LegendRenderOutput) -> Optional[bytes]:
    """Convert a LegendRenderOutput to raw image bytes.

        Args:
            lro: The legend render output object.

        Returns:
            The image encoded as bytes if available, otherwise None.
        """
    img = output_to_image(lro)
    return img.to_bytes() if img else None


def output_to_image(lro: gws.LegendRenderOutput) -> Optional[gws.Image]:
    """Extract an image object from a LegendRenderOutput.

        Args:
            lro: The legend render output object.

        Returns:
            The image object if available, otherwise None (e.g. when only HTML is set).
        """
    if lro.image:
        return lro.image
    if lro.image_path:
        return gws.lib.image.from_path(lro.image_path)
    if lro.html:
        return None


def output_to_image_path(lro: gws.LegendRenderOutput) -> Optional[str]:
    """Resolve the file path to the image of a LegendRenderOutput.

       Args:
           lro: The legend render output object.

       Returns:
           Path to the image file if available, otherwise None.
       """
    if lro.image:
        img_path = gws.u.ephemeral_path('legend.png')
        return lro.image.to_path(img_path, gws.lib.mime.PNG)
    if lro.image_path:
        return lro.image_path
    if lro.html:
        return None


def combine_outputs(lros: list[gws.LegendRenderOutput], options: dict = None) -> Optional[gws.LegendRenderOutput]:
    """Combine multiple LegendRenderOutputs into a single output.

        Args:
            lros: A list of legend render outputs to combine.
            options: Optional combination settings (currently unused).

        Returns:
            A new LegendRenderOutput containing the combined image,
            or None if no images were provided.
        """
    imgs = gws.u.compact(output_to_image(lro) for lro in lros)
    img = _combine_images(imgs, options)
    if not img:
        return None
    return gws.LegendRenderOutput(image=img, size=img.size())


def _combine_images(images: list[gws.Image], options: dict = None) -> Optional[gws.Image]:
    if not images:
        return None
    # @TODO other combination options
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
