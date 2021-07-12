import gws
import gws.types as t
import gws.lib.style
import gws.lib.shape
import gws.lib.svg
import gws.lib.xml2


_COMBINED_UID_DELIMITER = '___'


def from_geojson(js, crs, key_column='id') -> gws.IFeature:
    atts = {}
    uid = ''
    for k, v in js.get('properties', {}).items():
        if k == key_column:
            uid = v
        else:
            atts[k] = v

    return new(
        uid=uid,
        attributes=atts,
        shape=gws.lib.shape.from_geometry(js['geometry'], crs),
    )


def from_props(p: gws.Data) -> gws.IFeature:
    return new(
        uid=p.get('uid'),
        attributes=p.get('attributes'),
        elements=p.get('elements'),
        shape=p.get('shape'),
        style=p.get('style'),
    )


def new(uid, attributes=None, elements=None, shape=None, style=None):
    if isinstance(uid, str) and _COMBINED_UID_DELIMITER in uid:
        uid = uid.split(_COMBINED_UID_DELIMITER)[-1]

    obj = Feature()

    obj.uid = uid
    obj.elements = elements or {}

    obj.attributes = []
    if attributes:
        if isinstance(attributes, dict):
            attributes = [gws.Attribute({'name': k, 'value': v}) for k, v in attributes.items()]
        obj.attributes = attributes

    if shape:
        if isinstance(shape, gws.lib.shape.Shape):
            obj.shape = shape
        else:
            obj.shape = gws.lib.shape.from_props(shape)

    if style:
        obj.style = gws.lib.style.from_props(gws.Props(style))

    return obj


class Props(gws.Props):
    attributes: t.Optional[t.List[gws.Attribute]]
    elements: t.Optional[dict]
    layerUid: t.Optional[str]
    shape: t.Optional[gws.lib.shape.Props]
    style: t.Optional[gws.lib.style.Props]
    uid: t.Optional[str]



class Feature(gws.Object, gws.IFeature):

    def __init__(self):
        super().__init__()

        self.attributes = []
        self.category = ''
        self.elements = {}
        self.uid = ''

        self.layer: t.Optional[gws.ILayer] = None
        self.shape: t.Optional[gws.IShape] = None
        self.style: t.Optional[gws.IStyle] = None
        self.templates: t.Optional[gws.ITemplateBundle] = None
        self.data_model: t.Optional[gws.IDataModel] = None

    @property
    def props(self):
        return gws.Props(
            uid=self.full_uid,
            attributes=self.attributes,
            shape=self.shape.props if self.shape else None,
            style=self.style,
            elements=self.elements,
            layerUid=self.layer.uid if self.layer else None,
        )

    @property
    def props_for_render(self):
        return gws.Props(
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

    def transform_to(self, crs) -> gws.IFeature:
        if self.shape:
            self.shape = self.shape.transformed_to(crs)
        return self

    def to_svg_tags(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> t.List[gws.Tag]:
        if not self.shape:
            return []
        style = self.style or style
        if not style and self.layer:
            style = self.layer.style
        shape = self.shape.transformed_to(rv.bounds.crs)
        return gws.lib.svg.geometry_tags(
            t.cast(gws.lib.shape.Shape, shape).geom,
            rv,
            style.data if style else None,
            self.elements.get('label', ''))

    def to_svg(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> str:
        return gws.lib.svg.as_xml(self.to_svg_tags(rv, style))

    def to_geojson(self) -> dict:
        props = {a.name: a.value for a in self.attributes}
        props['id'] = self.uid
        return {
            'type': 'Feature',
            'properties': props,
            'geometry': self.shape.props.get('geometry') if self.shape else None
        }

    def apply_data_model(self, model: gws.IDataModel = None) -> gws.IFeature:
        model = model or self.data_model
        if model:
            self.attributes = model.apply(self.attributes)
        return self

    def apply_templates(self, templates: gws.ITemplateBundle = None, extra_context: dict = None, keys: t.List[str] = None) -> gws.IFeature:
        templates = templates or self.templates
        if templates:
            used = set()
            ctx = gws.merge(self.template_context, extra_context)
            for tpl in templates.all():
                if tpl.category == 'feature' and (tpl.key not in used) and (not keys or tpl.key in keys):
                    self.elements[tpl.key] = tpl.render(context=ctx).content
                    used.add(tpl.key)
        return self
