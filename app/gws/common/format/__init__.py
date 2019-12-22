import gws
import gws.types as t

_TPL_KEYS = [
    'category',
    'description',
    'label',
    'teaser',
    'title',
]


class Object(gws.Object, t.FormatObject):
    def configure(self):
        super().configure()

        for key in _TPL_KEYS:
            p = self.var(key)
            if p:
                setattr(self, key, self.add_child('gws.ext.template', p))

    def apply(self, feature: t.Feature, context: dict = None):
        ctx = gws.extend(
            {'feature': feature, 'attributes': feature.attributes},
            context,
            feature.attributes)
        for key in _TPL_KEYS:
            tpl = getattr(self, key, None)
            if tpl:
                res = tpl.render(dict(ctx)).content
                setattr(feature, key, res)

        return self
