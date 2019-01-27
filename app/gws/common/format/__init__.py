import gws
import gws.types as t

_TPL_KEYS = [
    'category',
    'description',
    'label',
    'teaser',
    'title',
]


class Object(gws.PublicObject, t.FormatInterface):
    def __init__(self):
        super().__init__()
        self.data_model = {}

        self.category: t.TemplateObject = None
        self.description: t.TemplateObject = None
        self.label: t.TemplateObject = None
        self.teaser: t.TemplateObject = None
        self.title: t.TemplateObject = None

    def configure(self):
        super().configure()

        for key in _TPL_KEYS:
            p = self.var(key)
            if p:
                setattr(self, key, self.add_child('gws.ext.template', p))

        self.data_model = self.var('dataModel')

    def apply(self, feature: t.FeatureInterface, context: dict = None):
        for key in _TPL_KEYS:
            tpl = getattr(self, key, None)
            if tpl:
                ctx = gws.extend({'feature': feature, 'attributes': feature.attributes}, context)
                res = tpl.render(ctx).content
                setattr(feature, key, res)

        dm = self.data_model
        if dm:
            feature.attributes = self.apply_data_model(feature.attributes, dm)

        return self

    def apply_data_model(self, data, data_model):
        # @TODO merge with printer
        d = {}
        for attr in data_model:
            if attr.name in data:
                # @TODO convert to type
                d[attr.name] = gws.as_str(data[attr.name])

        return d
