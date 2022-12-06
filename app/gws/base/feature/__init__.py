import gws
import gws.base.shape
import gws.lib.style
import gws.lib.svg
import gws.types as t


class Props(gws.Props):
    attributes: dict
    elements: dict
    shape: t.Optional[gws.base.shape.Props]
    uid: str
    modelUid: str


class Feature(gws.IFeature):
    def __init__(self, model, uid):
        self.uid = uid
        self.model = model

        self.attributes = {}
        self.elements = {}

        self.shape = None
        self.layerName = ''

        self.isNew = False

    def props(self, user):
        return Props(
            uid=self.uid,
            attributes={},
            modelUid=self.model.uid,
            shape=self.shape,
            elements=self.elements,
        )

    def apply_template(self, template, extra_args=None):
        tri = gws.TemplateRenderInput(args=gws.merge(self.attributes, extra_args, feature=self))
        key = template.subject.split('.')[-1]
        self.elements[key] = template.render(tri).content
        return self

    def attr(self, name, default=None):
        return self.attributes.get(name, default)

    def transform_to(self, crs) -> gws.IFeature:
        if self.shape:
            self.shape = self.shape.transformed_to(crs)
        return self

    def to_svg_fragment(self, view, style=None):
        if not self.shape:
            return []
        shape = self.shape.transformed_to(view.bounds.crs)
        return gws.lib.svg.shape_to_fragment(shape, view, style, self.elements.get('label'))

    def to_geojson(self):
        ps = dict(self.attributes)
        ps['id'] = self.uid
        d = {'type': 'Feature', 'properties': ps}
        if self.shape:
            d['geometry'] = self.shape.to_geojson()
            d['crs'] = self.shape.crs.to_geojson()
        return d
