"""Static legend.

Uses an image file as the legend.

Example::

    legend {
        type "static"
        path "/data/legend.png"
    }
"""

import gws
import gws.lib.image
import gws.base.legend


@gws.ext.config.legend('static')
class Config(gws.base.legend.Config):
    """Legend from a static image file."""

    path: gws.FilePath
    """Path to the legend image file."""


@gws.ext.object.legend('static')
class Object(gws.base.legend.Object):
    """Static legend."""

    path: str
    """Path to the image file."""

    def configure(self):
        self.path = self.cfg('path')

    def render(self, args=None):
        img = gws.lib.image.from_path(self.path)
        return gws.LegendRenderOutput(image=img, size=img.size())
