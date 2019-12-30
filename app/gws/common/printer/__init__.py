import gws
import gws.types as t


class Config(t.Config):
    templates: t.List[t.ext.template.Config]


class Props(t.Data):
    templates: t.List[t.TemplateProps]


#:export IPrinter
class Object(gws.Object, t.IPrinter):
    def __init__(self):
        super().__init__()
        self.templates: t.List[t.ITemplate] = []

    def configure(self):
        super().configure()
        self.templates = [
            self.add_child('gws.ext.template', p)
            for p in self.var('templates')
        ]

    @property
    def props(self) -> Props:
        return Props({
            'templates': self.templates,
        })
