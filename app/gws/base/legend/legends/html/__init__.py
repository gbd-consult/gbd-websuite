import gws
import gws.lib.image
import gws.lib.mime
import gws.types as t

from ... import core


@gws.ext.config.legend('html')
class Config(core.Config):
    """HTML-based legend."""

    template: gws.ext.config.template 
    """template for an HTML legend"""


@gws.ext.object.legend('html')
class Object(core.Object):
    template: gws.ITemplate

    def configure(self):
        self.template = self.create_child(gws.ext.object.template, self.var('template'))

    def render(self, args=None):
        # @TODO return html legends as html
        res = self.template.render(gws.TemplateRenderInput(
            args=args,
            out_mime=gws.lib.mime.PNG))
        img = gws.lib.image.from_path(res.path)
        return gws.LegendRenderOutput(image=img, size=img.size())
