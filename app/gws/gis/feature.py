import gws
import gws.types as t
import gws.gis.shape
import gws.common.style
import gws.tools.svg
import gws.tools.xml2

_COMBINED_UID_DELIMITER = '___'




# def from_geojson(js, crs, key_name='id'):
#     rec = dict(js.get('properties', {}))
#     rec['geometry'] = gws.gis.shape.from_geometry(js['geometry'], crs)
#     f = Feature()
#     f.attributes = rec
#     f.key_name = key_name
#     return f


#:export
class FeatureError(t.Data):
    fieldName: str
    message: str


#:export
class FeatureProps(t.Data):
    attributes: t.Optional[dict]
    category: t.Optional[str]
    elements: t.Optional[dict]
    errors: t.Optional[t.List[dict]]
    geometryName: t.Optional[str]
    isNew: t.Optional[bool]
    keyName: t.Optional[str]
    layerUid: t.Optional[str]
    modelUid: t.Optional[str]
    style: t.Optional[t.StyleProps]
    type: t.Optional[str]
    uid: t.Optional[str]


#:export IFeature
class Feature(t.IFeature):
    def __init__(self, model: t.IModel):
        self.model: t.IModel = model

        self.attributes: dict = {}
        self.category: str = ''
        self.elements: dict = {}
        self.geometry_name: str = ''
        self.key_name: str = ''
        self.layer: t.Optional[t.ILayer] = None
        self.style: t.Optional[t.IStyle] = None
        self.is_new: bool = False
        self.errors: t.List[t.FeatureError] = []

    def get_view_props(self, user: t.IUser) -> t.FeatureProps:
        mc = t.ModelContext(user=user, depth=1)
        fp = self.model.feature_props(self, mc)
        fp.attributes = {
            k: v
            for k, v in fp.attributes.items()
            if k in {self.key_name, self.geometry_name}
        }
        # del fp.modelUid
        return fp

    def get_props(self, user: t.IUser) -> t.FeatureProps:
        mc = t.ModelContext(user=user, depth=1)
        return self.model.feature_props(self, mc)

    @property
    def template_context(self) -> dict:
        d = dict(self.attributes)
        d['category'] = self.category
        d['feature'] = self
        d['layer'] = self.layer
        d['uid'] = self.uid
        return d

    @property
    def uid(self) -> t.Any:
        return self.attributes.get(self.key_name)

    def get_uid(self):
        return self.attributes.get(self.key_name)

    @uid.setter
    def uid(self, val):
        self.attributes[self.key_name] = val

    @property
    def shape(self) -> t.IShape:
        return self.attributes.get(self.geometry_name)

    def attr(self, name: str):
        return self.attributes.get(name)

    def transform_to(self, crs) -> t.IFeature:
        if self.shape:
            self.attributes[self.geometry_name] = self.shape.transformed_to(crs)
        return self

    def to_svg_tags(self, rv: t.MapRenderView, style: t.IStyle = None) -> t.List[t.Tag]:
        if not self.shape:
            return []
        if not style and self.layer:
            style = self.layer.style
        shape = self.shape.transformed_to(rv.bounds.crs)
        return gws.tools.svg.geometry_tags(
            t.cast(gws.gis.shape.Shape, shape).geom,
            rv,
            style.values,
            self.elements.get('label', ''))

    def to_svg(self, rv: t.MapRenderView, style: t.IStyle = None) -> str:
        return gws.tools.svg.as_xml(self.to_svg_tags(rv, style))

    def to_geojson(self) -> dict:
        props = dict(self.attributes)
        props['id'] = self.uid
        return {
            'type': 'Feature',
            'properties': props,
            'geometry': self.shape.props.geometry if self.shape else None
        }

    def apply_template(self, key, templates: t.List[t.ITemplate] = None, extra_context: dict = None) -> t.IFeature:
        if not templates and self.layer:
            templates = self.layer.templates
        if not templates:
            return self

        ctx = self.template_context
        if extra_context:
            ctx = gws.merge(ctx, extra_context)

        for tpl in templates:
            if tpl.category == 'feature' and tpl.key == key:
                self.elements[key] = tpl.render(context=ctx).content
                return self

        return self
