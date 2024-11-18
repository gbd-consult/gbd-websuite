"""Static legend."""

import gws
import gws.lib.image
import gws.lib.mime
import gws.base.legend


gws.ext.new.legend('static')


class Config(gws.base.legend.Config):
    """Static legend."""

    path: gws.FilePath
    """path to the image file"""


class Object(gws.base.legend.Object):
    path: str

    def configure(self):
        self.path = self.cfg('path')

    def render(self, args=None):
        img = gws.lib.image.from_path(self.path)
        return gws.LegendRenderOutput(image=img, size=img.size(), mime=gws.lib.mime.for_path(self.path))
