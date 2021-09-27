import gws
import gws.types as t
import gws.base.template


class Config(gws.Config):
    """Printer configuration"""

    templates: t.List[gws.ext.template.Config]  #: print templates


class Props(gws.Data):
    templates: gws.base.template.BundleProps


class Object(gws.Object):
    templates: gws.base.template.Bundle

    @property
    def props(self):
        return Props(templates=self.templates)

    def configure(self):
        self.templates = gws.base.template.create_bundle(self, self.var('templates'))
