import re
import shapely.ops
import shapely.geometry
import svgis
from PIL import ImageFont

import gws
import gws.tools.misc as misc

DEFAULT_FONT_SIZE = 10
DEFAULT_MARK_SIZE = 10
DEFAULT_POINT_SIZE = 10


def _to_pixel(geo, bbox, dpi, scale):
    def trans(x, y):
        cx = x - bbox[0]
        cy = bbox[3] - y

        cx /= scale
        cy /= scale

        cx = misc.mm2px(cx * 1000, dpi)
        cy = misc.mm2px(cy * 1000, dpi)

        return cx, cy

    return shapely.ops.transform(trans, geo)


def _tag(name, attrs, content=None):
    attrs = ' '.join(f'{k}="{v}"' for k, v in gws.compact(attrs).items())
    if content:
        return f'<{name} {attrs}>{content}</{name}>'
    return f'<{name} {attrs}/>'


_all_props = (
    'fill',

    'label-anchor',
    'label-background',
    'label-fill',
    'label-font-family',
    'label-font-size',
    'label-font-style',
    'label-font-weight',
    'label-line-height',
    'label-offset-x',
    'label-offset-y',
    'label-padding',
    'label-placement',

    'mark',
    'mark-fill',
    'mark-size',
    'mark-stroke',
    'mark-stroke-dasharray',
    'mark-stroke-dashoffset',
    'mark-stroke-linecap',
    'mark-stroke-linejoin',
    'mark-stroke-miterlimit',
    'mark-stroke-width',

    'stroke',
    'stroke-dasharray',
    # 'stroke-dashoffset',
    # 'stroke-linecap',
    # 'stroke-linejoin',
    # 'stroke-miterlimit',
    'stroke-width',

    'point-size',

)

_color_patterns = (
    r'^#[0-9a-fA-F]{6}$',
    r'^#[0-9a-fA-F]{8}$',
    r'^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$',
    r'^rgba\(\d{1,3},\d{1,3},\d{1,3},\d?(\.\d{1,3})?\)$',
    r'^[a-z]{3,50}$',
)


def _as_color(val):
    if val is None:
        return None
    val = re.sub(r'\s+', '', val)
    if any(re.match(p, val) for p in _color_patterns):
        return val


def _as_px(val):
    if val is None:
        return None
    if isinstance(val, int):
        return val
    m = re.match(r'^(-?\d+)px', str(val))
    if m:
        return int(m.group(1))


def _as_px_array(val):
    if val is None:
        return None
    val = str(val).split()
    p = [None]

    if len(val) == 1:
        p = _as_px(val[0])
        p = [p, p, p, p]
    elif len(val) == 2:
        p = _as_px(val[0])
        q = _as_px(val[1])
        p = [p, q, p, q]
    elif len(val) == 4:
        p = [_as_px(q) for q in val]

    if all(q is not None for q in p):
        return p


def _as_enum(val, *args):
    if val is None:
        return None
    if any(a == val for a in args):
        return val


def _fill_stroke(attrs, css, prefix=''):
    v = _as_color(css.get(prefix + 'fill'))
    attrs['fill'] = v or 'none'

    v = _as_color(css.get(prefix + 'stroke'))
    if not v:
        return
    attrs['stroke'] = v

    v = _as_px(css.get(prefix + 'stroke-width'))
    attrs['stroke-width'] = f'{v or 1}px'

    v = css.get(prefix + 'stroke-dasharray')
    if v:
        v = v.split()
        if all(p.isdigit() for p in v):
            attrs['stroke-dasharray'] = ' '.join(v)


def _marker(uid, css):
    mark = css.get('mark')
    if not mark:
        return

    size = _as_px(css.get('mark-size')) or DEFAULT_MARK_SIZE
    size2 = size // 2

    content = None
    content_attrs = {}

    _fill_stroke(content_attrs, css, 'mark-')

    if mark == 'circle':
        content_attrs.update({
            'cx': size2,
            'cy': size2,
            'r': size2,
        })
        content = _tag('circle', content_attrs)

    return _tag('marker', {
        'id': uid,
        'viewBox': f'0 0 {size} {size}',
        'refX': size2,
        'refY': size2,
        'markerUnits': 'userSpaceOnUse',
        'markerWidth': size,
        'markerHeight': size,
    }, content)


def _map_font(css):
    # @TODO: allow for more fonts and customize the mapping

    DEFAULT_FONT = 'DejaVuSans'

    if css.get('label-font-weight') == 'bold':
        return DEFAULT_FONT + '-Bold'
    return DEFAULT_FONT


def _text(cx, cy, label, css):
    # @TODO label positioning needs more work

    font_name = _map_font(css)
    font_size = _as_px(css.get('label-font-size')) or DEFAULT_FONT_SIZE
    font = ImageFont.truetype(font=font_name, size=font_size)

    attrs = gws.compact({
        'font-family': font_name.split('-')[0],
        'font-size': f'{font_size}px',
        'font-weight': _as_enum(css.get('label-font-weight'), 'bold') or 'normal',
        'font-style': _as_enum(css.get('label-font-style'), 'italic') or 'normal',
        'text-anchor': _as_enum(css.get('label-anchor'), 'start', 'middle', 'end') or 'middle',
    })

    _fill_stroke(attrs, css, 'label-')

    lines = list(gws.lines(label))
    _, em_height = font.getsize('MMM')
    metrics = [font.getsize(s) for s in lines]

    line_height = _as_px(css.get('label-line-height')) or 1
    padding = _as_px_array(css.get('label-padding')) or [0, 0, 0, 0]

    ly = cy - padding[2]
    lx = cx

    anchor = attrs['text-anchor']

    if anchor == 'start':
        lx += padding[3]
    elif anchor == 'end':
        lx -= padding[1]
    else:
        lx += padding[3] // 2

    height = em_height * len(lines) + line_height * (len(lines) - 1) + padding[0] + padding[2]

    pad_bottom = metrics[-1][1] - em_height
    if pad_bottom > 0:
        height += pad_bottom
    ly -= pad_bottom

    spans = []
    for s in reversed(lines):
        spans.append(_tag('tspan', {'x': lx, 'y': ly}, s))
        ly -= (em_height + line_height)

    text = _tag('text', attrs, ''.join(reversed(spans)))
    bgr = _as_color(css.get('label-background'))

    if not bgr:
        return text

    width = max(xy[0] for xy in metrics) + padding[1] + padding[3]

    if anchor == 'start':
        bx = cx
    elif anchor == 'end':
        bx = cx - width
    else:
        bx = cx - width // 2

    rattrs = {
        'x': bx,
        'y': cy - height,
        'width': width,
        'height': height,
        'fill': bgr
    }

    return _tag('rect', rattrs) + text


def _points(geo):
    t = geo.type

    if t in ('Point', 'LineString', 'LinearRing'):
        return geo.coords
    if t in ('Polygon', 'LineString', 'LinearRing'):
        return geo.exterior.coords
    if t.startswith('Multi'):
        return [p for g in geo.geoms for p in _points(g)]


def _label_position(geo, css):
    placement = _as_enum(css.get('label-placement'), 'start', 'middle', 'end')
    if placement == 'start':
        x, y = _points(geo)[0]
    elif placement == 'end':
        x, y = _points(geo)[-1]
    else:
        c = geo.centroid
        x, y = c.x, c.y
    return (
        round(x) + (_as_px(css.get('label-offset-x')) or 0),
        round(y) + (_as_px(css.get('label-offset-y')) or 0),
    )


def _label(geo, label, css):
    cx, cy = _label_position(geo, css)
    return _text(cx, cy, label, css)


def draw(geo, label, style, bbox, dpi, scale, rotation):
    geo = _to_pixel(geo, bbox, dpi, scale)

    # @TODO use geo.svg
    if not style:
        return svgis.draw.geometry(shapely.geometry.mapping(geo), precision=0)

    # @TODO other style types
    css = style.content

    mark = ''
    mark_id = ''
    text = ''

    if 'mark' in css:
        mark_id = '_M' + gws.random_string(8)
        mark = _marker(mark_id, css)

    if label:
        text = _label(geo, label, css)

    attrs = {
        'precision': 0
    }

    _fill_stroke(attrs, css, '')

    if mark:
        attrs['marker-start'] = attrs['marker-mid'] = attrs['marker-end'] = f'url(#{mark_id})'

    if geo.type in ('Point', 'MultiPoint'):
        attrs['r'] = _as_px(css.get('point-size') or DEFAULT_POINT_SIZE) // 2

    if geo.type in ('LineString', 'MultiLineString'):
        attrs['fill'] = 'none'

    g = svgis.draw.geometry(shapely.geometry.mapping(geo), **gws.compact(attrs))
    return mark + g + text
