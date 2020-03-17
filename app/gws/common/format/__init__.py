import gws
import gws.types as t


#:export IFormat
class Object(gws.Object, t.IFormat):
    def configure(self):
        super().configure()

        self.templates = {}
        for key, p in self.config.as_dict().items():
            if p:
                self.templates[key] = self.add_child('gws.ext.template', p)

    def apply(self, context: dict) -> dict:
        res = {}
        for key, tpl in self.templates.items():
            res[key] = tpl.render(dict(context)).content
        return res
