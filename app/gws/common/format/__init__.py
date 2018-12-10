import gws
import gws.types as t


class Object(gws.PublicObject, t.FormatInterface):
    def __init__(self):
        super().__init__()
        self.description: t.TemplateObject = None
        self.category = ''
        self.label = ''
        self.data_model = {}
        self.teaser = None
        self.title = ''

    def configure(self):
        super().configure()

        p = self.var('description')
        if p:
            self.description = self.add_child('gws.ext.template', p)
        p = self.var('teaser')
        if p:
            self.teaser = self.add_child('gws.ext.template', p)
