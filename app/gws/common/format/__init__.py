import gws
import gws.types as t


#:export IFormat
class Object(gws.Object, t.IFormat):
    def __init__(self):
        super().__init__()
        self.templates = {}

    def configure(self):
        super().configure()

        for key, p in self.config.as_dict().items():
            if p:
                self.templates[key] = self.add_child('gws.ext.template', p)

    def apply(self, context: dict) -> dict:
        res = {}
        for key, tpl in self.templates.items():
            res[key] = tpl.render(dict(context)).content
        return res
