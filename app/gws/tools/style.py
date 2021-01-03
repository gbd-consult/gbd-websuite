import re
import base64

import gws
import gws.tools.net

import gws.types as t


#:export
class StyleStrokeLineCap(t.Enum):
    butt = 'butt'
    round = 'round'
    square = 'square'


#:export
class StyleStrokeLineJoin(t.Enum):
    bevel = 'bevel'
    round = 'round'
    miter = 'miter'


#:export
class StyleMarker(t.Enum):
    circle = 'circle'
    square = 'square'
    arrow = 'arrow'
    cross = 'cross'


#:export
class StyleLabelOption(t.Enum):
    all = 'all'
    none = 'none'


#:export
class StyleGeometryOption(t.Enum):
    all = 'all'
    none = 'none'


#:export
class StyleLabelAlign(t.Enum):
    left = 'left'
    right = 'right'
    center = 'center'


#:export
class StyleLabelPlacement(t.Enum):
    start = 'start'
    end = 'end'
    middle = 'middle'


#:export
class StyleLabelFontStyle(t.Enum):
    normal = 'normal'
    italic = 'italic'


#:export
class StyleLabelFontWeight(t.Enum):
    normal = 'normal'
    bold = 'bold'


#:export
class StyleValues(t.Data):
    fill: t.Optional[t.Color]

    stroke: t.Optional[t.Color]
    stroke_dasharray: t.Optional[t.List[int]]
    stroke_dashoffset: t.Optional[int]
    stroke_linecap: t.Optional[StyleStrokeLineCap]
    stroke_linejoin: t.Optional[StyleStrokeLineJoin]
    stroke_miterlimit: t.Optional[int]
    stroke_width: t.Optional[int]

    marker: t.Optional[StyleMarker]
    marker_fill: t.Optional[t.Color]
    marker_size: t.Optional[int]
    marker_stroke: t.Optional[t.Color]
    marker_stroke_dasharray: t.Optional[t.List[int]]
    marker_stroke_dashoffset: t.Optional[int]
    marker_stroke_linecap: t.Optional[StyleStrokeLineCap]
    marker_stroke_linejoin: t.Optional[StyleStrokeLineJoin]
    marker_stroke_miterlimit: t.Optional[int]
    marker_stroke_width: t.Optional[int]

    with_geometry: t.Optional[StyleGeometryOption]
    with_label: t.Optional[StyleLabelOption]

    label_align: t.Optional[StyleLabelAlign]
    label_background: t.Optional[t.Color]
    label_fill: t.Optional[t.Color]
    label_font_family: t.Optional[str]
    label_font_size: t.Optional[int]
    label_font_style: t.Optional[StyleLabelFontStyle]
    label_font_weight: t.Optional[StyleLabelFontWeight]
    label_line_height: t.Optional[int]
    label_max_scale: t.Optional[int]
    label_min_scale: t.Optional[int]
    label_offset_x: t.Optional[int]
    label_offset_y: t.Optional[int]
    label_padding: t.Optional[t.List[int]]
    label_placement: t.Optional[StyleLabelPlacement]
    label_stroke: t.Optional[t.Color]
    label_stroke_dasharray: t.Optional[t.List[int]]
    label_stroke_dashoffset: t.Optional[int]
    label_stroke_linecap: t.Optional[StyleStrokeLineCap]
    label_stroke_linejoin: t.Optional[StyleStrokeLineJoin]
    label_stroke_miterlimit: t.Optional[int]
    label_stroke_width: t.Optional[int]

    point_size: t.Optional[int]
    icon: t.Optional[str]

    offset_x: t.Optional[int]
    offset_y: t.Optional[int]


##

_DEFAULT_VALUES = dict(
    fill=None,

    stroke=None,
    stroke_dasharray=[],
    stroke_dashoffset=0,
    stroke_linecap=StyleStrokeLineCap.butt,
    stroke_linejoin=StyleStrokeLineJoin.miter,
    stroke_miterlimit=0,
    stroke_width=0,

    marker_size=0,
    marker_stroke_dasharray=[],
    marker_stroke_dashoffset=0,
    marker_stroke_linecap=StyleStrokeLineCap.butt,
    marker_stroke_linejoin=StyleStrokeLineJoin.miter,
    marker_stroke_miterlimit=0,
    marker_stroke_width=0,

    with_geometry=StyleGeometryOption.all,
    with_label=StyleLabelOption.all,

    label_align=StyleLabelAlign.center,
    label_font_family='sans-serif',
    label_font_size=12,
    label_font_style=StyleLabelFontStyle.normal,
    label_font_weight=StyleLabelFontWeight.normal,
    label_line_height=1,
    label_max_scale=1000000000,
    label_min_scale=0,
    label_offset_x=0,
    label_offset_y=0,
    label_placement=StyleLabelPlacement.middle,
    label_stroke_dasharray=[],
    label_stroke_dashoffset=0,
    label_stroke_linecap=StyleStrokeLineCap.butt,
    label_stroke_linejoin=StyleStrokeLineJoin.miter,
    label_stroke_miterlimit=0,
    label_stroke_width=0,

    point_size=10,
    icon='',

    offset_x=0,
    offset_y=0,
)

##

_color_patterns = (
    r'^#[0-9a-fA-F]{3}$',
    r'^#[0-9a-fA-F]{6}$',
    r'^#[0-9a-fA-F]{8}$',
    r'^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$',
    r'^rgba\(\d{1,3},\d{1,3},\d{1,3},\d?(\.\d{1,3})?\)$',
    r'^[a-z]{3,50}$',
)


def _color(val):
    val = re.sub(r'\s+', '', str(val))
    if any(re.match(p, val) for p in _color_patterns):
        return val


def _icon(val):
    if not val:
        return
    s = val
    m = re.match(r'^url\((.+?)\)$', val)
    if m:
        s = m.group(1).strip('\'\"')
    try:
        return _to_data_url(s)
    except Exception as e:
        raise gws.Error(f'cannot load {val!r}') from e


def _to_data_url(val):
    if val.startswith('data:'):
        return val
    # @TODO security, this should be only allowed in a trusted context
    if re.match(r'^https?:', val):
        svg = gws.tools.net.http_request(val).content
    else:
        svg = gws.read_file_b(val)
    return 'data:image/svg+xml;base64,' + base64.standard_b64encode(svg).decode('utf8')


def _px(val):
    if isinstance(val, int):
        return val
    m = re.match(r'^(-?\d+)px', str(val))
    return _int(m.group(1) if m else val)


def _int(val):
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except:
        pass


def _intlist(val):
    if not isinstance(val, (list, tuple)):
        val = re.split(r'[,\s]+', str(val))
    val = [_int(x) for x in val]
    if any(x is None for x in val):
        return None
    return val


def _padding(val):
    if not isinstance(val, (list, tuple)):
        val = re.split(r'[,\s]+', str(val))
    val = [_px(x) for x in val]
    if any(x is None for x in val):
        return None
    if len(val) == 4:
        return val
    if len(val) == 2:
        return [val[0], val[1], val[0], val[1]]
    if len(val) == 1:
        return [val[0], val[0], val[0], val[0]]
    return None


def _enum(cls):
    def _check(val):
        if val in vars(cls):
            return val

    return _check


def _str(val):
    val = str(val).strip()
    return val or None


##

class _Parser:
    fill = _color

    stroke = _color
    stroke_dasharray = _intlist
    stroke_dashoffset = _px
    stroke_linecap = _enum(StyleStrokeLineCap)
    stroke_linejoin = _enum(StyleStrokeLineJoin)
    stroke_miterlimit = _px
    stroke_width = _px

    marker = _enum(StyleMarker)
    marker_fill = _color
    marker_size = _px
    marker_stroke = _color
    marker_stroke_dasharray = _intlist
    marker_stroke_dashoffset = _px
    marker_stroke_linecap = _enum(StyleStrokeLineCap)
    marker_stroke_linejoin = _enum(StyleStrokeLineJoin)
    marker_stroke_miterlimit = _px
    marker_stroke_width = _px

    with_geometry = _enum(StyleGeometryOption)
    with_label = _enum(StyleLabelOption)

    label_align = _enum(StyleLabelAlign)
    label_background = _color
    label_fill = _color
    label_font_family = _str
    label_font_size = _px
    label_font_style = _enum(StyleLabelFontStyle)
    label_font_weight = _enum(StyleLabelFontWeight)
    label_line_height = _int
    label_max_scale = _int
    label_min_scale = _int
    label_offset_x = _px
    label_offset_y = _px
    label_padding = _padding
    label_placement = _enum(StyleLabelPlacement)
    label_stroke = _color
    label_stroke_dasharray = _intlist
    label_stroke_dashoffset = _px
    label_stroke_linecap = _enum(StyleStrokeLineCap)
    label_stroke_linejoin = _enum(StyleStrokeLineJoin)
    label_stroke_miterlimit = _px
    label_stroke_width = _px

    point_size = _px
    icon = _icon

    offset_x = _px
    offset_y = _px


def from_css_dict(d: dict) -> t.StyleValues:
    values = t.StyleValues(_DEFAULT_VALUES)

    for k, v in d.items():
        if v is None:
            continue
        k = k.replace('-', '_')
        if k.startswith('__'):
            k = k[2:]
        fn = gws.get(_Parser, k)
        if fn:
            v = fn(v)
            if v is not None:
                setattr(values, k, v)

    return values


# @TODO use a real parser
def from_css_text(text) -> t.StyleValues:
    d = {}
    for r in text.split(';'):
        r = r.strip()
        if not r:
            continue
        r = r.split(':')
        d[r[0].strip()] = r[1].strip()
    return from_css_dict(d)


def parse_icon(val):
    return _icon(val)
