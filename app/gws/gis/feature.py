import gws
import gws.types as t
import gws.gis.shape
import gws.gis.svg


def from_geojs(js, crs):
    return Feature({
        'attributes': js.get('properties'),
        'shape': gws.gis.shape.from_geometry(js['geometry'], crs),
    })


def from_attributes(attrs):
    return Feature({
        'attributes': attrs,
    })


def from_props(p: t.FeatureProps):
    sh = None
    if p.get('shape'):
        sh = gws.gis.shape.from_props(p.get('shape'))

    style = None
    if p.get('style'):
        style = p.get('style')
        if isinstance(style, dict):
            style = t.StyleProps(style)

    return Feature({
        'uid': p.get('uid'),
        'attributes': p.get('attributes'),
        'label': p.get('label'),
        'shape': sh,
        'style': style
    })


class Feature(t.Feature):
    def __init__(self, args):
        self.attributes = {}
        self.description = ''
        self.category = ''
        self.label = ''
        self.shape = None
        self.style = None
        self.teaser = ''
        self.title = ''
        self.uid = ''

        for k, v in gws.as_dict(args).items():
            setattr(self, k, v)

        s = args.get('shape')
        if s:
            if isinstance(s, t.Shape):
                self.shape = s
            elif s.get('geometry'):
                self.shape = gws.gis.shape.from_props(s)

        s = args.get('style')
        if s:
            if isinstance(s, dict):
                s = t.StyleProps(s)
            self.style = s

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

    def to_svg(self, bbox, dpi, scale, rotation):
        if not self.shape:
            return ''
        return gws.gis.svg.draw(
            self.shape.geo,
            self.label,
            self.style,
            bbox,
            dpi,
            scale,
            rotation
        )

    def to_geojs(self):
        geometry = None
        if self.shape:
            geometry = self.shape.props['geometry']
        return {
            'type': 'Feature',
            'properties': self.attributes,
            'geometry': geometry
        }

    def apply_format(self, fmt: t.FormatObject, context: dict = None):
        if fmt:
            fmt.apply(self, context)

    @property
    def props(self):
        return t.FeatureProps({
            'attributes': self.attributes or {},
            'category': self.category,
            'description': self.description,
            'label': self.label,
            'shape': self.shape.props if self.shape else None,
            'style': self.style,
            'teaser': self.teaser,
            'title': self.title,
            'uid': self.uid,
        })
