import base64
import re

import gws
import gws.types as t
import gws.lib.net


class Type(t.Enum):
    css = 'css'
    cssSelector = 'cssSelector'


class Values(gws.Data):
    fill: gws.Color

    stroke: gws.Color
    stroke_dasharray: t.List[int]
    stroke_dashoffset: int
    stroke_linecap: t.Literal['butt', 'round', 'square']
    stroke_linejoin: t.Literal['bevel', 'round', 'miter']
    stroke_miterlimit: int
    stroke_width: int

    marker: t.Literal['circle', 'square', 'arrow', 'cross']
    marker_fill: gws.Color
    marker_size: int
    marker_stroke: gws.Color
    marker_stroke_dasharray: t.List[int]
    marker_stroke_dashoffset: int
    marker_stroke_linecap: t.Literal['butt', 'round', 'square']
    marker_stroke_linejoin: t.Literal['bevel', 'round', 'miter']
    marker_stroke_miterlimit: int
    marker_stroke_width: int

    with_geometry: t.Literal['all', 'none']
    with_label: t.Literal['all', 'none']

    label_align: t.Literal['left', 'right', 'center']
    label_background: gws.Color
    label_fill: gws.Color
    label_font_family: str
    label_font_size: int
    label_font_style: t.Literal['normal', 'italic']
    label_font_weight: t.Literal['normal', 'bold']
    label_line_height: int
    label_max_scale: int
    label_min_scale: int
    label_offset_x: int
    label_offset_y: int
    label_padding: t.List[int]
    label_placement: t.Literal['start', 'end', 'middle']
    label_stroke: gws.Color
    label_stroke_dasharray: t.List[int]
    label_stroke_dashoffset: int
    label_stroke_linecap: t.Literal['butt', 'round', 'square']
    label_stroke_linejoin: t.Literal['bevel', 'round', 'miter']
    label_stroke_miterlimit: int
    label_stroke_width: int

    point_size: int
    icon: str

    offset_x: int
    offset_y: int


class Data(gws.StyleData):
    name: str
    type: Type
    text: str
    values: Values


class Config(gws.Config):
    """Feature style"""

    type: Type  #: style type
    name: t.Optional[str]  #: style name
    text: t.Optional[str]  #: raw style content
    values: t.Optional[dict]  #: style values


class Props(gws.Props):
    type: Type
    values: dict
    text: str = ''
    name: str = ''


class Object(gws.Node, gws.IStyle):
    data: gws.StyleData

    @property
    def props(self):
        d = t.cast(Data, self.data)
        return gws.Props(
            type=d.type,
            values=vars(d.values),
            text=d.text or '',
            name=d.name or '')


def from_config(cfg: Config) -> Object:
    return from_props(t.cast(gws.Props, cfg))


def from_props(p: gws.Props) -> Object:
    typ = p.get('type', 'css')
    data = gws.StyleData(p, type=typ)

    if typ == 'css':
        val = p.get('values')
        if val:
            data.set('values', values_from_dict(gws.as_dict(val)))
        else:
            data.set('values', values_from_text(p.get('text')))

    obj = Object()
    obj.data = data
    return obj


def values_from_dict(d: dict) -> Values:
    values = Values(_DEFAULT_VALUES)

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
def values_from_text(text) -> Values:
    d = {}
    for r in text.split(';'):
        r = r.strip()
        if not r:
            continue
        r = r.split(':')
        d[r[0].strip()] = r[1].strip()
    return values_from_dict(d)


def parse_icon(val):
    return _icon(val)


_DEFAULT_VALUES = gws.Data(
    fill=None,

    stroke=None,
    stroke_dasharray=[],
    stroke_dashoffset=0,
    stroke_linecap='butt',
    stroke_linejoin='miter',
    stroke_miterlimit=0,
    stroke_width=0,

    marker_size=0,
    marker_stroke_dasharray=[],
    marker_stroke_dashoffset=0,
    marker_stroke_linecap='butt',
    marker_stroke_linejoin='miter',
    marker_stroke_miterlimit=0,
    marker_stroke_width=0,

    with_geometry='all',
    with_label='all',

    label_align='center',
    label_font_family='sans-serif',
    label_font_size=12,
    label_font_style='normal',
    label_font_weight='normal',
    label_line_height=1,
    label_max_scale=1000000000,
    label_min_scale=0,
    label_offset_x=0,
    label_offset_y=0,
    label_placement='middle',
    label_stroke_dasharray=[],
    label_stroke_dashoffset=0,
    label_stroke_linecap='butt',
    label_stroke_linejoin='miter',
    label_stroke_miterlimit=0,
    label_stroke_width=0,

    point_size=10,
    icon='',

    offset_x=0,
    offset_y=0,
)

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
        svg = gws.lib.net.http_request(val).content
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


class _Parser:
    fill = _color

    stroke = _color
    stroke_dasharray = _intlist
    stroke_dashoffset = _px
    stroke_linecap = _enum('stroke_linecap')
    stroke_linejoin = _enum('stroke_linejoin')
    stroke_miterlimit = _px
    stroke_width = _px

    marker = _enum('marker')
    marker_fill = _color
    marker_size = _px
    marker_stroke = _color
    marker_stroke_dasharray = _intlist
    marker_stroke_dashoffset = _px
    marker_stroke_linecap = _enum('stroke_linecap')
    marker_stroke_linejoin = _enum('stroke_linejoin')
    marker_stroke_miterlimit = _px
    marker_stroke_width = _px

    with_geometry = _enum('x')
    with_label = _enum('x')

    label_align = _enum('x')
    label_background = _color
    label_fill = _color
    label_font_family = _str
    label_font_size = _px
    label_font_style = _enum('x')
    label_font_weight = _enum('x')
    label_line_height = _int
    label_max_scale = _int
    label_min_scale = _int
    label_offset_x = _px
    label_offset_y = _px
    label_padding = _padding
    label_placement = _enum('x')
    label_stroke = _color
    label_stroke_dasharray = _intlist
    label_stroke_dashoffset = _px
    label_stroke_linecap = _enum('x')
    label_stroke_linejoin = _enum('x')
    label_stroke_miterlimit = _px
    label_stroke_width = _px

    point_size = _px
    icon = _icon

    offset_x = _px
    offset_y = _px
