import gws
import gws.base.template
import gws.types as t


class Config(t.Config):
    """Printer configuration"""

    templates: t.List[t.ext.template.Config] #: print templates


class Props(t.Data):
    templates: t.List[gws.base.template.TemplateProps]


#:export IPrinter
class Object(gws.Object, t.IPrinter):
    def configure(self):
        super().configure()
        self.templates: t.List[t.ITemplate] = gws.base.template.bundle(self, self.var('templates'))

    @property
    def props(self) -> Props:
        return Props({
            'templates': self.templates,
        })
