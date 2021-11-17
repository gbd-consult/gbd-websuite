import gws
import gws.base.template
import gws.types as t


class Config(gws.Config):
    """Printer configuration"""

    templates: t.List[gws.ext.template.Config]  #: print templates


class Props(gws.Data):
    templates: gws.base.template.bundle.Props


class Object(gws.Node):
    templates: gws.base.template.bundle.Object

    def props_for(self, user):
        return gws.Data(templates=self.templates)

    def configure(self):
        self.templates = gws.base.template.bundle.create(
            self.root,
            items=self.var('templates'),
            parent=self)
