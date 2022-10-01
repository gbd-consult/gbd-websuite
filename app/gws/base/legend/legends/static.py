import gws
import gws.lib.image
import gws.types as t

from .. import main


@gws.ext.config.legend('static')
class Config(main.Config):
    """Static legend."""

    path: gws.FilePath  #: path to the image file


@gws.ext.object.legend('static')
class Object(main.Object):
    path: str

    def configure(self):
        self.path = self.var('path')

    def render(self, args=None):
        img = gws.lib.image.from_path(self.path)
        return gws.LegendRenderOutput(image=img, size=img.size)
