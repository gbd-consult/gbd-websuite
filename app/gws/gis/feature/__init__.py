import gws
import gws.gis.shape
import gws.lib.style
import gws.lib.svg
import gws.types as t


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
        shape=gws.gis.shape.from_geometry(js['geometry'], crs),
    )


def from_args(**kwargs) -> gws.IFeature:
    return Feature(**kwargs)


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
    shape: t.Optional[gws.gis.shape.Props]
    style: t.Optional[gws.lib.style.Props]
    uid: t.Optional[str]


_COMBINED_UID_DELIMITER = '___'


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
            if isinstance(shape, gws.gis.shape.Shape):
                self.shape = shape
            else:
                self.shape = gws.gis.shape.from_props(shape)

        if style:
            self.style = gws.lib.style.from_props(gws.Props(style))

    def props_for(self, user):
        return gws.Data(
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

    def to_svg_fragment(self, view, style=None):
        if not self.shape:
            return []
        style = style or self.style or (self.layer.style if self.layer else None)
        shape = self.shape.transformed_to(view.bounds.crs)
        return gws.lib.svg.shape_to_fragment(shape, view, style, self.elements.get('label'))

    def to_svg_element(self, view, style=None):
        fr = self.to_svg_fragment(view, style)
        if fr:
            return gws.lib.svg.fragment_to_element(fr)

    def to_geojson(self):
        ps = {a.name: a.value for a in self.attributes}
        ps['id'] = self.uid
        d = {'type': 'Feature', 'properties': ps}
        if self.shape:
            d['geometry'] = self.shape.to_geojson()
            d['crs'] = self.shape.crs.to_geojson()
        return d

    def connect_to(self, layer):
        self.layer = layer
        return self

    def apply_data_model(self, model=None):
        model = model or self.data_model or (self.layer.data_model if self.layer else None)
        if model:
            self.attributes = model.apply(self.attributes)
        return self

    def apply_templates(self, templates=None, extra_context=None, subjects=None):
        templates = templates or self.templates or (self.layer.templates if self.layer else None)
        if not templates:
            return self

        used = set()
        ctx = gws.merge(self.template_context, extra_context)

        if subjects:
            templates = [tpl for tpl in templates.items if tpl.subject in subjects]
        else:
            templates = [tpl for tpl in templates.items if tpl.category == 'feature']

        for tpl in templates:
            if tpl.name not in used:
                self.elements[tpl.name] = tpl.render().content
                used.add(tpl.name)

        return self
