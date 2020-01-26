import re

import gws
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
    stroke_miterLimit: t.Optional[int]
    stroke_width: t.Optional[int]

    marker: t.Optional[StyleMarker]
    marker_fill: t.Optional[t.Color]
    marker_size: t.Optional[int]
    marker_stroke: t.Optional[t.Color]
    marker_stroke_width: t.Optional[int]

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
    label_stroke_width: t.Optional[int]
    label_width: t.Optional[int]

    point_size: t.Optional[int]


##

_color_patterns = (
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


def _px(val):
    if isinstance(val, int):
        return val
    m = re.match(r'^(-?\d+)px', str(val))
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    return _int(val)


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
    stroke_miterLimit = _px
    stroke_width = _px

    marker = _enum(StyleMarker)
    marker_fill = _color
    marker_size = _px
    marker_stroke = _color
    marker_stroke_width = _px

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
    label_stroke_width = _px
    label_width = _px

    point_size = _px


def from_css_dict(d: dict) -> t.StyleValues:
    values = t.StyleValues()

    for k, v in d.items():
        if v is None:
            continue
        k = k.replace('-', '_')
        if k.startswith('__'):
            k = k[2:]
        fn = getattr(_Parser, k, None)
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
