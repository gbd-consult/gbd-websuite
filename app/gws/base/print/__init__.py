import gws
import gws.types as t
import gws.base.template



class Config(gws.Config):
    """Printer configuration"""

    templates: t.List[gws.ext.template.Config]  #: print templates


class Props(gws.Data):
    templates: t.List[gws.base.template.Props]


class Object(gws.Node):
    templates: gws.base.template.Bundle

    @property
    def props(self):
        return Props(templates=self.templates)

    def configure(self):
        p = self.var('templates')
        self.templates = t.cast(gws.base.template.Bundle, self.create_child(
            gws.base.template.Bundle,
            gws.Config(templates=p)))
