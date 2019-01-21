import gws
import gws.types as t
import gws.gis.shape
import gws.gis.svg
import gws.tools.misc as misc


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


class Feature(t.FeatureInterface):
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
            if isinstance(s, t.ShapeInterface):
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

    def apply_format(self, fmt: t.FormatInterface):
        s = fmt.label
        if s:
            self.label = gws.tools.misc.format_placeholders(s, self.attributes)

        s = fmt.category
        if s:
            self.category = gws.tools.misc.format_placeholders(s, self.attributes)

        s = fmt.title
        if s:
            self.title = gws.tools.misc.format_placeholders(s, self.attributes)

        s = fmt.description
        if s:
            self.description = s.render({'feature': self, 'attributes': self.attributes}).content

        s = fmt.teaser
        if s:
            self.teaser = s.render({'feature': self, 'attributes': self.attributes}).content

        s = fmt.data_model
        if s:
            self.attributes = self._validate_data(self.attributes, s)

        return self

    def _validate_data(self, data, data_model):
        # @TODO merge with printer
        d = {}
        for attr in data_model:
            if attr.name in data:
                # @TODO convert to type
                d[attr.name] = gws.as_str(data[attr.name])

        return d


    @property
    def props(self):
        return t.FeatureProps({
            'attributes': self.attributes or {},
            'description': self.description,
            'label': self.label,
            'shape': self.shape.props if self.shape else None,
            'style': self.style,
            'teaser': self.teaser,
            'title': self.title,
            'uid': self.uid,
        })
