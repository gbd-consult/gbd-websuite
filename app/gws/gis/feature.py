import gws
import gws.types as t
import gws.gis.shape
import gws.common.style
import gws.tools.svg
import gws.tools.xml2

_COMBINED_UID_DELIMITER = '___'


def from_geojson(js, crs, key_column='id'):
    atts = {}
    uid = ''
    for k, v in js.get('properties', {}).items():
        if k == key_column:
            uid = v
        else:
            atts[k] = v

    return Feature(
        uid=uid,
        attributes=atts,
        shape=gws.gis.shape.from_geometry(js['geometry'], crs),
    )


def from_props(p: t.FeatureProps):
    return Feature(
        uid=p.get('uid'),
        attributes=p.get('attributes'),
        elements=p.get('elements'),
        shape=p.get('shape'),
        style=p.get('style'),
    )


#:export
class FeatureProps(t.Data):
    uid: t.Optional[str]
    attributes: t.Optional[t.List[t.Attribute]]
    elements: t.Optional[dict]
    layerUid: t.Optional[str]
    shape: t.Optional[t.ShapeProps]
    style: t.Optional[t.StyleProps]


#:export IFeature
class Feature(t.IFeature):
    def __init__(self, uid=None, attributes=None, category=None, elements=None, shape=None, style=None):
        self.attributes: t.List[t.Attribute] = []
        self.category: str = category
        self.feature_format: t.Optional[t.IFormat] = None
        self.data_model: t.Optional[t.IModel] = None
        self.elements = {}
        self.layer: t.Optional[t.ILayer] = None
        self.shape: t.Optional[t.IShape] = None
        self.style: t.Optional[t.IStyle] = None
        self.uid: str = ''

        self._init(uid, attributes, elements, shape, style)

    @property
    def props(self) -> t.FeatureProps:
        return t.FeatureProps(
            uid=self.full_uid,
            attributes=self.attributes,
            shape=self.shape.props if self.shape else None,
            style=self.style,
            elements=self.elements,
            layerUid=self.layer.uid if self.layer else None,
        )

    @property
    def props_for_render(self) -> t.FeatureProps:
        return t.FeatureProps(
            uid=self.full_uid,
            attributes=[],
            shape=self.shape.props if self.shape else None,
            layerUid=self.layer.uid if self.layer else None,
        )

    @property
    def full_uid(self) -> str:
        uid = self.uid or ''
        if self.layer:
            uid = f'{self.layer.uid}{_COMBINED_UID_DELIMITER}{uid}'
        return uid

    @property
    def template_context(self) -> dict:
        d = {a.name: a.value for a in self.attributes}
        d['category'] = self.category
        d['feature'] = self
        d['layer'] = self.layer
        d['uid'] = self.uid
        return d

    @property
    def attr_dict(self) -> dict:
        return {a.name: a.value for a in self.attributes}

    def attr(self, name: str):
        for a in self.attributes:
            if a.name == name:
                return a.value

    def transform_to(self, crs) -> t.IFeature:
        if self.shape:
            self.shape = self.shape.transformed_to(crs)
        return self

    def to_svg_tags(self, rv: t.MapRenderView, style: t.IStyle = None) -> t.List[t.Tag]:
        if not self.shape:
            return []
        style = self.style or style
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
        props = {a.name: a.value for a in self.attributes}
        props['id'] = self.uid
        return {
            'type': 'Feature',
            'properties': props,
            'geometry': self.shape.props.geometry if self.shape else None
        }

    def apply_data_model(self, model: t.IModel = None) -> t.IFeature:
        model = model or self.data_model
        if model:
            self.attributes = model.apply(self.attributes)
        return self

    def apply_format(self, fmt: t.IFormat = None, extra_context: dict = None, keys: t.List[str] = None) -> t.IFeature:
        fmt = fmt or self.feature_format
        if fmt:
            self.elements = gws.merge(
                self.elements,
                fmt.apply(gws.merge(self.template_context, extra_context), keys))
        return self

    def _init(self, uid, attributes, elements, shape, style):
        if isinstance(uid, str) and _COMBINED_UID_DELIMITER in uid:
            uid = uid.split(_COMBINED_UID_DELIMITER)[-1]

        self.uid = uid
        self.elements = elements or {}

        self.attributes = []
        if attributes:
            if isinstance(attributes, dict):
                attributes = [t.Attribute({'name': k, 'value': v}) for k, v in attributes.items()]
            self.attributes = attributes

        if shape:
            if isinstance(shape, gws.gis.shape.Shape):
                self.shape = shape
            else:
                self.shape = gws.gis.shape.from_props(shape)

        if style:
            if isinstance(style, dict):
                style = t.StyleProps(style)
            self.style = gws.common.style.from_props(style)
