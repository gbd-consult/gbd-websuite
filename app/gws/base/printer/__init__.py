import gws
import gws.types as t
import gws.base.template


class Config(gws.Config):
    """Printer configuration"""

    templates: t.List[gws.ext.template.Config]  #: print templates


class Props(gws.Data):
    templates: gws.base.template.bundle.Props


class Object(gws.Node):
    templates: gws.base.template.bundle.Object

    def props_for(self, user):
        return Props(templates=self.templates)

    def configure(self):
        self.templates = gws.base.template.bundle.create(
            self.root,
            gws.Config(templates=self.var('templates')),
            parent=self)
