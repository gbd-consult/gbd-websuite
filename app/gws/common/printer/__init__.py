import gws
import gws.types as t


class Config(t.Config):
    templates: t.List[t.ext.template.Config]


class Props(t.Data):
    templates: t.List[t.TemplateProps]


#:export IPrinter
class Object(gws.Object, t.IPrinter):
    def configure(self):
        super().configure()
        self.templates: t.List[t.ITemplate] = [
            self.create_child('gws.ext.template', p)
            for p in self.var('templates')
        ]

    @property
    def props(self) -> Props:
        return Props({
            'templates': self.templates,
        })
