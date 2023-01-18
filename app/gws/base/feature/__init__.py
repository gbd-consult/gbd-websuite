import gws
import gws.base.shape
import gws.lib.style
import gws.lib.svg
import gws.types as t


class Props(gws.Props):
    attributes: dict
    views: dict
    uid: str
    keyName: str
    geometryName: str
    isNew: bool
    modelUid: str


def with_model(model: gws.IModel):
    return Feature(model)


class Feature(gws.Object, gws.IFeature):
    def __init__(self, model):
        self.model = model
        self.attributes = {}
        self.views = {}
        self.errors = []

        self.isNew = False

    def props(self, user):
        return self.model.feature_props(self, user)

    def uid(self):
        if self.model.keyName:
            return self.attributes.get(self.model.keyName)

    def shape(self):
        if self.model.geometryName:
            return self.attributes.get(self.model.geometryName)

    def attr(self, name, default=None):
        return self.attributes.get(name, default)

    def compute_values(self, access, user, **kwargs):
        self.model.compute_values(self, access, user, **kwargs)
        return self

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
        return gws.lib.svg.shape_to_fragment(shape, view, style, label)

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
