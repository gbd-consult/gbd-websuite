import base64
import re
import gws.lib.net
import gws
import gws.types as t


def parse_dict(d: dict) -> dict:
    res = dict(_DEFAULTS)

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
                res[k] = v

    return res


# @TODO use a real CSS parser

def parse_text(text: str) -> dict:
    d = {}
    for r in text.split(';'):
        r = r.strip()
        if not r:
            continue
        a, _, b = r.partition(':')
        d[a.strip()] = b.strip()
    return parse_dict(d)


def parse_icon(val) -> t.Optional[gws.Url]:
    return _icon(val)


##


_DEFAULTS: dict = dict(
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

_ENUMS = dict(
    stroke_linecap=['butt', 'round', 'square'],
    stroke_linejoin=['bevel', 'round', 'miter'],
    marker=['circle', 'square', 'arrow', 'cross'],
    marker_stroke_linecap=['butt', 'round', 'square'],
    marker_stroke_linejoin=['bevel', 'round', 'miter'],
    with_geometry=['all', 'none'],
    with_label=['all', 'none'],
    label_align=['left', 'right', 'center'],
    label_font_style=['normal', 'italic'],
    label_font_weight=['normal', 'bold'],
    label_padding=[int],
    label_placement=['start', 'end', 'middle'],
    label_stroke_dasharray=[int],
    label_stroke_linecap=['butt', 'round', 'square'],
    label_stroke_linejoin=['bevel', 'round', 'miter'],
)

_COLOR_PATTERNS = (
    r'^#[0-9a-fA-F]{3}$',
    r'^#[0-9a-fA-F]{6}$',
    r'^#[0-9a-fA-F]{8}$',
    r'^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$',
    r'^rgba\(\d{1,3},\d{1,3},\d{1,3},\d?(\.\d{1,3})?\)$',
    r'^[a-z]{3,50}$',
)


##

def _color(val):
    val = re.sub(r'\s+', '', str(val))
    if any(re.match(p, val) for p in _COLOR_PATTERNS):
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
    if re.match(r'^https?:', val):
        # @TODO security, this should be only allowed in a trusted context
        raise gws.Error(f'cannot load remote {val!r}')
        # svg = gws.lib.net.http_request(val).content
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
        vals = _ENUMS.get(cls)
        if vals and val in vals:
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
