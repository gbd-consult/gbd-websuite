import gws
import gws.types as t
import gws.gis.shape
import gws.gis.svg


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
        shape=p.get('shape'),
        style=p.get('style'),
    )


def new(args: dict):
    uid = args.pop('uid', None)
    attributes = args.pop('attributes', None)
    shape = args.pop('shape', None)
    style = args.pop('style', None)
    elements = args
    return Feature(uid, attributes, elements, shape, style)


class Feature(t.Feature):
    def __init__(self, uid=None, attributes=None, elements=None, shape=None, style=None):
        self.uid = uid
        self.elements = elements or {}
        self.convertor = None
        self.layer = None

        self.attributes = []
        if attributes:
            if isinstance(attributes, dict):
                self.attributes = [t.Attribute({'name': k, 'value': v}) for k, v in attributes.items()]
            elif isinstance(attributes, (list, tuple)):
                for a in attributes:
                    if not isinstance(a, t.Data):
                        a = t.Attribute(a)
                    self.attributes.append(a)

        self.shape = None
        if shape:
            if isinstance(shape, t.Shape):
                self.shape = shape
            elif shape.get('geometry'):
                self.shape = gws.gis.shape.from_props(shape)

        self.style = None
        if style:
            if isinstance(style, dict):
                style = t.StyleProps(style)
            self.style = style

    def transform(self, to_crs):
        if self.shape:
            self.shape = self.shape.transform(to_crs)
        return self

    def set_default_style(self, style):
        if self.style or not style:
            return
        if isinstance(style, dict):
            style = t.StyleProps(style)
        self.style = style

    def to_svg(self, rv: t.RenderView, style: t.Style = None):
        if not self.shape:
            return ''
        style = self.style or style
        if not style and self.layer:
            style = self.layer.style
        return gws.gis.svg.draw(
            self.shape.geo,
            self.elements.get('label', ''),
            style,
            rv.bbox,
            rv.dpi,
            rv.scale,
            rv.rotation
        )

    def to_geojson(self):
        geometry = None
        if self.shape:
            geometry = self.shape.props['geometry']
        props = {a.name: a.value for a in self.attributes}
        props['id'] = self.uid
        return {
            'type': 'Feature',
            'properties': props,
            'geometry': geometry
        }

    def convert(self, target_crs: t.Crs = None, convertor: t.FeatureConvertor = None) -> 'Feature':
        if self.shape and target_crs:
            self.shape = self.shape.transform(target_crs)

        convertor = convertor or self.convertor or self.layer

        if convertor:
            if convertor.data_model:
                self.attributes = convertor.data_model.apply(self.attributes)

            if convertor.feature_format:
                ctx = gws.extend(
                    {'feature': self, 'layer': self.layer},
                    {a.name: a.value for a in self.attributes},
                    self.elements)
                self.elements = gws.extend(self.elements, convertor.feature_format.apply(ctx))

        return self

    @property
    def props(self):
        return t.FeatureProps({
            'uid': self.uid,
            'attributes': self.attributes,
            'shape': self.shape.props if self.shape else None,
            'style': self.style,
            'elements': self.elements,
            'layerUid': self.layer.uid if self.layer else None,
        })
