import gws
import gws.base.shape
import gws.lib.style
import gws.lib.svg
import gws.types as t


def new(
        model: gws.IModel,
        record: t.Optional[gws.FeatureRecord] = None,
        props: t.Optional[gws.FeatureProps] = None
) -> gws.IFeature:
    f = Feature(model)
    f.record = record or gws.FeatureRecord(attributes={})
    f.props = props or gws.FeatureProps(attributes={})
    return f


class Feature(gws.Object, gws.IFeature):
    def __init__(self, model: gws.IModel):
        self.attributes = {}
        self.cssSelector = ''
        self.errors = []
        self.isNew = False
        self.layerName = ''
        self.model = model
        self.views = {}
        self.createWithFeatures = []
        self.insertedPrimaryKey = ''

    def __repr__(self):
        try:
            return f'<feature {self.model.uid}:{self.uid()}>'
        except:
            return f'<feature ?>'

    def uid(self):
        if self.model.uidName:
            return str(self.attributes.get(self.model.uidName))
        return ''

    def shape(self):
        if self.model.geometryName:
            return self.attributes.get(self.model.geometryName)

    def get(self, name, default=None):
        return self.attributes.get(name, default)

    def has(self, name):
        return name in self.attributes

    def set(self, name, value):
        self.attributes[name] = value
        return self

    def raw(self, name):
        return self.record.attributes.get(name)

    def render_views(self, templates, **kwargs):
        tri = gws.TemplateRenderInput(
            args=gws.merge(
                self.attributes,
                kwargs,
                feature=self
            ))
        for tpl in templates:
            view_name = tpl.subject.split('.')[-1]
            self.views[view_name] = tpl.render(tri).content
        return self

    def transform_to(self, crs) -> gws.IFeature:
        if self.shape():
            self.attributes[self.model.geometryName] = self.shape().transformed_to(crs)
        return self

    def to_svg(self, view, label=None, style=None):
        if not self.shape():
            return []
        shape = self.shape().transformed_to(view.bounds.crs)
        return gws.lib.svg.shape_to_fragment(shape, view, label, style)

    def to_geojson(self, user):
        p = self.props(user)
        d = {'type': 'Feature', 'properties': getattr(p, 'attributes', {})}
        d['properties']['id'] = self.uid()

        if self.model.geometryName:
            shape = d['properties'].pop(self.model.geometryName, None)
            if shape:
                d['geometry'] = shape.to_geojson()
                d['crs'] = shape.crs.to_geojson()

        return d
