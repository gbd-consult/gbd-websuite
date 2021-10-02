import gws
import gws.types as t
import gws.base.template


class Config(gws.Config):
    """Printer configuration"""

    templates: t.List[gws.ext.template.Config]  #: print templates


class Props(gws.Data):
    templates: gws.base.template.bundle.Props


class Object(gws.Object):
    templates: gws.base.template.bundle.Object

    @property
    def props(self):
        return Props(templates=self.templates)

    def configure(self):
        self.templates = gws.base.template.bundle.create(
            self.root,
            gws.Config(templates=self.var('templates')),
            parent=self)
