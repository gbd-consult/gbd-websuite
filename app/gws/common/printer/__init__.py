import gws
import gws.types as t


class Config(t.Config):
    templates: t.List[t.ext.template.Config]


class Props(t.Data):
    templates: t.List[t.TemplateProps]


class Object(gws.Object):
    templates = []

    def configure(self):
        super().configure()
        self.templates = [
            self.add_child('gws.ext.template', p)
            for p in self.var('templates')
        ]

    @property
    def props(self):
        return Props({
            'templates': self.templates,
        })
