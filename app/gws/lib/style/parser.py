import base64
import re

import gws
import gws.lib.net
import gws.types as t
from . import icon


def parse_dict(d: dict, trusted, with_strict_mode) -> dict:
    res = dict(_DEFAULTS)

    for key, val in d.items():
        if val is None:
            continue
        k = key.replace('-', '_')
        if k.startswith('__'):
            k = k[2:]

        fn = gws.get(_ParseFunctions, k)

        if not fn:
            err = f'style: invalid css property {key!r}'
            if with_strict_mode:
                raise gws.Error(err)
            else:
                gws.log.error(err)
                continue

        try:
            v = fn(val, trusted)
            if v is not None:
                res[k] = v
        except Exception as exc:
            err = f'style: invalid css value for {key!r}'
            if with_strict_mode:
                raise gws.Error(err) from exc
            else:
                gws.log.error(err)

    return res


# @TODO use a real CSS parser

def parse_text(text: str, trusted, with_strict_mode) -> dict:
    d = {}
    for r in text.split(';'):
        r = r.strip()
        if not r:
            continue
        a, _, b = r.partition(':')
        d[a.strip()] = b.strip()
    return parse_dict(d, trusted, with_strict_mode)


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
    icon=None,
    parsed_icon=None,

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

def _parse_color(val, trusted):
    val = re.sub(r'\s+', '', str(val))
    if any(re.match(p, val) for p in _COLOR_PATTERNS):
        return val


def _parse_intlist(val, trusted):
    return [int(x) for x in _make_list(val)]


def _parse_icon(val, trusted):
    return icon.parse(val, trusted)


def _parse_unitint(val, trusted):
    return _unitint(val)


def _parse_unitintquad(val, trusted):
    val = [_unitint(x) for x in _make_list(val)]
    if any(x is None for x in val):
        return None
    if len(val) == 4:
        return val
    if len(val) == 2:
        return [val[0], val[1], val[0], val[1]]
    if len(val) == 1:
        return [val[0], val[0], val[0], val[0]]


def _parse_enum_fn(cls):
    def _check(val, trusted):
        vals = _ENUMS.get(cls)
        if vals and val in vals:
            return val

    return _check


def _parse_int(val):
    return int(val)


def _parse_str(val):
    val = str(val).strip()
    return val or None


##

def _unitint(val):
    if isinstance(val, int):
        return val
    m = re.match(r'^(-?\d+)([a-z]*)', str(val))
    if not m:
        return

    # @TODO support other units (need to pass dpi here)
    if m.group(2) != 'px':
        return

    return int(m.group(1))


def _make_list(val):
    if isinstance(val, (list, tuple)):
        return val
    return re.split(r'[,\s]+', str(val))


##


class _ParseFunctions:
    fill = _parse_color

    stroke = _parse_color
    stroke_dasharray = _parse_intlist
    stroke_dashoffset = _parse_unitint
    stroke_linecap = _parse_enum_fn('stroke_linecap')
    stroke_linejoin = _parse_enum_fn('stroke_linejoin')
    stroke_miterlimit = _parse_unitint
    stroke_width = _parse_unitint

    marker = _parse_enum_fn('marker')
    marker_fill = _parse_color
    marker_size = _parse_unitint
    marker_stroke = _parse_color
    marker_stroke_dasharray = _parse_intlist
    marker_stroke_dashoffset = _parse_unitint
    marker_stroke_linecap = _parse_enum_fn('stroke_linecap')
    marker_stroke_linejoin = _parse_enum_fn('stroke_linejoin')
    marker_stroke_miterlimit = _parse_unitint
    marker_stroke_width = _parse_unitint

    with_geometry = _parse_enum_fn('x')
    with_label = _parse_enum_fn('x')

    label_align = _parse_enum_fn('x')
    label_background = _parse_color
    label_fill = _parse_color
    label_font_family = _parse_str
    label_font_size = _parse_unitint
    label_font_style = _parse_enum_fn('x')
    label_font_weight = _parse_enum_fn('x')
    label_line_height = _parse_int
    label_max_scale = _parse_int
    label_min_scale = _parse_int
    label_offset_x = _parse_unitint
    label_offset_y = _parse_unitint
    label_padding = _parse_unitintquad
    label_placement = _parse_enum_fn('x')
    label_stroke = _parse_color
    label_stroke_dasharray = _parse_intlist
    label_stroke_dashoffset = _parse_unitint
    label_stroke_linecap = _parse_enum_fn('x')
    label_stroke_linejoin = _parse_enum_fn('x')
    label_stroke_miterlimit = _parse_unitint
    label_stroke_width = _parse_unitint

    point_size = _parse_unitint
    icon = _parse_icon

    offset_x = _parse_unitint
    offset_y = _parse_unitint
