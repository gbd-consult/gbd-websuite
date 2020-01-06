import gws
import gws.types as t
import gws.gis.shape
import gws.gis.svg

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


#:export
class FeatureConvertor:
    feature_format: t.IFormat
    data_model: t.IModel


#:export IFeature
class Feature(t.IFeature):
    def __init__(self, uid=None, attributes=None, category=None, elements=None, shape=None, style=None):
        self.attributes: t.List[t.Attribute] = []
        self.elements = {}
        self.category: str = category
        self.convertor: FeatureConvertor = None
        self.layer: t.ILayer = None
        self.shape: t.IShape = None
        self.style: t.IStyle = None
        self.uid: str = ''

        self._init(uid, attributes, elements, shape, style)

    @property
    def props(self) -> t.FeatureProps:
        uid = self.uid or ''
        if self.layer:
            uid = f'{self.layer.uid}{_COMBINED_UID_DELIMITER}{uid}'
        return t.FeatureProps({
            'uid': uid,
            'attributes': self.attributes,
            'shape': self.shape.props if self.shape else None,
            'style': self.style,
            'elements': self.elements,
            'layerUid': self.layer.uid if self.layer else None,
        })

    def transform(self, to_crs) -> t.IFeature:
        if self.shape:
            self.shape = self.shape.transformed(to_crs)
        return self

    def to_svg(self, rv: t.RenderView, style: t.IStyle = None) -> str:
        if not self.shape:
            return ''
        style = self.style or style
        if not style and self.layer:
            style = self.layer.style
        s: gws.gis.shape.Shape = self.shape.transformed(rv.bounds.crs)
        return gws.gis.svg.draw(
            s.geom,
            self.elements.get('label', ''),
            style,
            rv.bounds.extent,
            rv.dpi,
            rv.scale,
            rv.rotation
        )

    def to_geojson(self) -> dict:
        props = {a.name: a.value for a in self.attributes}
        props['id'] = self.uid
        return {
            'type': 'Feature',
            'properties': props,
            'geometry': self.shape.props.geometry if self.shape else None
        }

    def convert(self, target_crs: t.Crs = None, convertor: t.FeatureConvertor = None) -> t.IFeature:
        if self.shape and target_crs:
            self.shape = self.shape.transformed(target_crs)

        convertor = convertor or self.convertor or self.layer

        if convertor:
            if convertor.data_model:
                self.attributes = convertor.data_model.apply(self.attributes)

            atts = {a.name: a.value for a in self.attributes}

            if convertor.feature_format:
                ctx = gws.extend(
                    atts,
                    category=self.category,
                    feature=self,
                    layer=self.layer,
                    uid=self.uid,
                )
                self.elements = gws.extend(self.elements, convertor.feature_format.apply(ctx))

        return self

    def _init(self, uid, attributes, elements, shape, style):
        if isinstance(uid, str) and _COMBINED_UID_DELIMITER in uid:
            uid = uid.split(_COMBINED_UID_DELIMITER)[-1]

        self.uid = uid

        self.elements = elements or {}
        self.convertor = None
        self.layer = None

        self.attributes = []
        if attributes:
            if isinstance(attributes, dict):
                attributes = [t.Attribute({'name': k, 'value': v}) for k, v in attributes.items()]
            self.attributes = attributes

        self.shape = None
        if shape:
            if isinstance(shape, gws.gis.shape.Shape):
                self.shape = shape
            else:
                self.shape = gws.gis.shape.from_props(shape)

        self.style = None
        if style:
            if isinstance(style, dict):
                style = t.StyleProps(style)
            self.style = style
