import gws
import gws.common.template
import gws.types as t


class Config(t.Config):
    templates: t.List[t.ext.template.Config]


class Props(t.Data):
    templates: t.List[t.TemplateProps]


#:export IPrinter
class Object(gws.Object, t.IPrinter):
    def configure(self):
        super().configure()
        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'))

    @property
    def props(self) -> Props:
        return Props({
            'templates': self.templates,
        })
