"""HTML legend."""

import gws
import gws.base.legend
import gws.lib.image
import gws.lib.mime

gws.ext.new.legend('html')


class Config(gws.base.legend.Config):
    """HTML-based legend."""

    template: gws.ext.config.template
    """template for an HTML legend"""


class Object(gws.base.legend.Object):
    template: gws.ITemplate

    def configure(self):
        self.template = self.create_child(gws.ext.object.template, self.cfg('template'))

    def render(self, args=None):
        # @TODO return html legends as html
        res = self.template.render(gws.TemplateRenderInput(
            args=args,
            mimeOut=gws.lib.mime.PNG))
        img = gws.lib.image.from_path(res.contentPath)
        return gws.LegendRenderOutput(image=img, size=img.size())
