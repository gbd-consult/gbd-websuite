import gws
import gws.types as t


class Object(gws.PublicObject, t.FormatInterface):
    def __init__(self):
        super().__init__()
        self.category = ''
        self.data_model = {}
        self.description: t.TemplateObject = None
        self.label = ''
        self.teaser: t.TemplateObject = None
        self.title = ''

    def configure(self):
        super().configure()

        self.category = self.var('category')
        self.data_model = self.var('dataModel')
        self.label = self.var('label')
        self.title = self.var('title')

        p = self.var('description')
        if p:
            self.description = self.add_child('gws.ext.template', p)
        p = self.var('teaser')
        if p:
            self.teaser = self.add_child('gws.ext.template', p)
