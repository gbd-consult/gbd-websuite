import gws
import gws.lib.shape
import gws.lib.style
import gws.lib.svg
import gws.lib.xml2
import gws.types as t

_COMBINED_UID_DELIMITER = '___'


def from_geojson(js, crs, key_column='id') -> gws.IFeature:
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
        shape=gws.lib.shape.from_geometry(js['geometry'], crs),
    )


def from_props(p: gws.Data) -> gws.IFeature:
    return Feature(
        uid=p.get('uid'),
        attributes=p.get('attributes'),
        elements=p.get('elements'),
        shape=p.get('shape'),
        style=p.get('style'),
    )


class Props(gws.Props):
    attributes: t.Optional[t.List[gws.Attribute]]
    elements: t.Optional[dict]
    layerUid: t.Optional[str]
    shape: t.Optional[gws.lib.shape.Props]
    style: t.Optional[gws.lib.style.Props]
    uid: t.Optional[str]


class Feature(gws.Object, gws.IFeature):
    # attributes: t.List[gws.Attribute]
    # category: str
    # data_model: t.Optional[gws.IDataModel]
    # elements: dict
    # layer: t.Optional[gws.ILayer]
    # shape: t.Optional[gws.IShape]
    # style: t.Optional[gws.IStyle]
    # templates: t.Optional[gws.ITemplateBundle]
    # uid: str

    def __init__(self, uid, attributes=None, category=None, elements=None, shape=None, style=None):
        super().__init__()
        self.attributes = []
        self.category = category or ''
        self.data_model = None
        self.elements = {}
        self.layer = None
        self.shape = None
        self.style = None
        self.templates = None
        self.uid = ''

        if isinstance(uid, str) and _COMBINED_UID_DELIMITER in uid:
            uid = uid.split(_COMBINED_UID_DELIMITER)[-1]

        self.uid = gws.to_str(uid)

        if attributes:
            if isinstance(attributes, dict):
                attributes = [gws.Attribute({'name': k, 'value': v}) for k, v in attributes.items()]
            self.attributes = attributes

        if elements:
            self.elements = elements

        if shape:
            if isinstance(shape, gws.lib.shape.Shape):
                self.shape = shape
            else:
                self.shape = gws.lib.shape.from_props(shape)

        if style:
            self.style = gws.lib.style.from_props(gws.Props(style))

    def props_for(self, user):
        return Props(
            uid=self.full_uid,
            attributes=self.attributes,
            shape=self.shape,
            style=self.style,
            elements=self.elements,
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
        style = style or self.style or (self.layer.style if self.layer else None)
        shape = self.shape.transformed_to(rv.bounds.crs)
        return gws.lib.svg.geometry_tags(
            t.cast(gws.lib.shape.Shape, shape).geom,
            rv,
            t.cast(gws.lib.style.Values, style.values) if style else None,
            self.elements.get('label', ''))

    def to_svg(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> str:
        return gws.lib.svg.to_xml(self.to_svg_tags(rv, style))

    def to_geojson(self) -> dict:
        ps = {a.name: a.value for a in self.attributes}
        ps['id'] = self.uid
        return {
            'type': 'Feature',
            'properties': ps,
            'geometry': self.shape.to_geojson() if self.shape else None
        }

    def connect_to(self, layer):
        self.layer = layer
        return self

    def apply_data_model(self, model: gws.IDataModel = None) -> gws.IFeature:
        model = model or self.data_model or (self.layer.data_model if self.layer else None)
        if model:
            self.attributes = model.apply(self.attributes)
        return self

    def apply_templates(self, templates: gws.ITemplateBundle = None, extra_context: dict = None, template_names: t.List[str] = None) -> gws.IFeature:
        templates = templates or self.templates or (self.layer.templates if self.layer else None)
        if templates:
            used = set()
            ctx = gws.merge(self.template_context, extra_context)
            for tpl in templates.items:
                if tpl.category == 'feature' and (tpl.name not in used) and (not template_names or tpl.name in template_names):
                    self.elements[tpl.name] = tpl.render(context=ctx).content
                    used.add(tpl.name)
        return self
