import re
import shapely.ops
import shapely.geometry
import svgis
from PIL import ImageFont
import bs4
import wand.image

import gws
import gws.tools.units as units

import gws.types as t

DEFAULT_FONT_SIZE = 10
DEFAULT_MARKER_SIZE = 10
DEFAULT_POINT_SIZE = 10


def draw(geom, label: str, sv: t.StyleValues, extent: t.Extent, dpi: int, scale: int, rotation: int) -> str:
    geom = _to_pixel(geom, extent, dpi, scale)

    # @TODO use geom.svg
    if not sv:
        return svgis.draw.geometry(shapely.geometry.mapping(geom), precision=0)

    marker = ''
    marker_id = ''
    text = ''
    icon = ''

    if sv.marker:
        marker_id = '_M' + gws.random_string(8)
        marker = _marker(marker_id, sv)

    if sv.icon:
        ico = _parse_icon(sv.icon, dpi)
        if not ico:
            gws.log.warn(f'cannot parse icon {sv.icon!r}')
        else:
            svg, w, h = ico
            x, y = _icon_position(geom, svg, w, h)
            svg['x'] = f'{x}px'
            svg['y'] = f'{y}px'
            icon = str(svg)

    extra_y_offset = 0

    if sv.label_offset_y is None:
        if geom.type == 'Point':
            extra_y_offset = 12
        if geom.type == 'LineString':
            extra_y_offset = 6

    # @TODO with_label, scale
    if label:
        text = _label(geom, label, sv, extra_y_offset)

    atts = {
        'precision': 0
    }

    _fill_stroke(atts, sv, '')

    if marker:
        atts['marker-start'] = atts['marker-mid'] = atts['marker-end'] = f'url(#{marker_id})'

    if geom.type in ('Point', 'MultiPoint'):
        atts['r'] = (sv.point_size or DEFAULT_POINT_SIZE) // 2

    if geom.type in ('LineString', 'MultiLineString'):
        atts['fill'] = 'none'

    g = svgis.draw.geometry(shapely.geometry.mapping(geom), **gws.compact(atts))
    return marker + g + icon + text


def to_png(elements, size: t.Size):
    elements = '\n'.join(elements)
    svg = f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg">{elements}</svg>'

    with wand.image.Image(blob=svg.encode('utf8'), format='svg', background='None', width=size[0], height=size[1]) as image:
        return image.make_blob('png')


def convert_fragment(svg_fragment, rv: t.RenderView):
    def trans(x, y):
        cx = x - rv.bounds.extent[0]
        cy = rv.bounds.extent[3] - y

        cx /= rv.scale
        cy /= rv.scale

        cx = units.mm2px(cx * 1000, rv.dpi)
        cy = units.mm2px(cy * 1000, rv.dpi)

        return cx, cy

    def repl(m):
        pt = points[int(m.group(1))]
        n = pt[int(m.group(2))]
        n += int(m.group(3))
        return str(n)

    """
        svg fragments contain point array (coordinates)
        and an svg string, with <<...>> placeholders
        which are <<point-number  0=x,1=y  offset>>
    """

    points = [trans(x, y) for x, y in svg_fragment.points]

    svg = re.sub(r'<<(\S+) (\S+) (\S+)>>', repl, svg_fragment.svg)

    return svg


##

def _to_pixel(geom, extent, dpi, scale):
    def trans(x, y):
        cx = x - extent[0]
        cy = extent[3] - y

        cx /= scale
        cy /= scale

        cx = units.mm2px(cx * 1000, dpi)
        cy = units.mm2px(cy * 1000, dpi)

        return cx, cy

    return shapely.ops.transform(trans, geom)


def _marker(uid, sv):
    size = sv.marker_size or DEFAULT_MARKER_SIZE
    size2 = size // 2

    content = None
    atts = {}

    _fill_stroke(atts, sv, 'marker_')

    if sv.marker == 'circle':
        atts.update({
            'cx': size2,
            'cy': size2,
            'r': size2,
        })
        content = _tag('circle', atts)

    if content:
        return _tag('marker', {
            'id': uid,
            'viewBox': f'0 0 {size} {size}',
            'refX': size2,
            'refY': size2,
            'markerUnits': 'userSpaceOnUse',
            'markerWidth': size,
            'markerHeight': size,
        }, content)


def _label(geom, label, sv, extra_y_offset=0):
    cx, cy = _label_position(geom, sv, extra_y_offset)
    return _text(cx, cy, label, sv)


def _label_position(geom, sv, extra_y_offset=0):
    if sv.label_placement == 'start':
        x, y = _get_points(geom)[0]
    elif sv.label_placement == 'end':
        x, y = _get_points(geom)[-1]
    else:
        c = geom.centroid
        x, y = c.x, c.y
    return (
        round(x) + (sv.label_offset_x or 0),
        round(y) + extra_y_offset + (sv.label_font_size >> 1) + (sv.label_offset_y or 0)
    )


_PFX_SVF_UTF8 = 'data:image/svg+xml;utf8,'


def _parse_icon(s, dpi):
    content = None

    if s.startswith(_PFX_SVF_UTF8):
        content = s[len(_PFX_SVF_UTF8):]

    if not content:
        return

    # @TODO
    svg = bs4.BeautifulSoup(content, 'lxml-xml').svg
    if not svg:
        return

    w = svg.attrs.get('width')
    h = svg.attrs.get('height')

    if not w or not h:
        return

    try:
        w, wu = units.parse(w, units=['px', 'mm'], default='px')
        h, hu = units.parse(h, units=['px', 'mm'], default='px')
    except ValueError:
        return

    if wu == 'mm':
        w = units.mm2px(w, dpi)
    if hu == 'mm':
        h = units.mm2px(h, dpi)

    svg['width'] = f'{w}px'
    svg['height'] = f'{h}px'

    return svg, w, h


def _icon_position(geom, sv, width, height):
    c = geom.centroid
    return c.x - (int(width) >> 1), c.y - (int(height) >> 1)


def _get_points(geom):
    gt = geom.type

    if gt in ('Point', 'LineString', 'LinearRing'):
        return geom.coords
    if gt in ('Polygon', 'LineString', 'LinearRing'):
        return geom.exterior.coords
    if gt.startswith('Multi'):
        return [p for g in geom.geoms for p in _get_points(g)]


def _text(cx, cy, label, sv):
    # @TODO label positioning needs more work

    font_name = _map_font(sv)
    font_size = sv.label_font_size or DEFAULT_FONT_SIZE
    font = ImageFont.truetype(font=font_name, size=font_size)

    anchor = 'start'

    if sv.label_align == 'right':
        anchor = 'end'
    elif sv.label_align == 'center':
        anchor = 'middle'

    atts = gws.compact({
        'font-family': font_name.split('-')[0],
        'font-size': f'{font_size}px',
        'font-weight': sv.label_font_weight,
        'font-style': sv.label_font_style,
        'text-anchor': anchor,
    })

    _fill_stroke(atts, sv, 'label_')

    lines = list(gws.lines(label))
    _, em_height = font.getsize('MMM')
    metrics = [font.getsize(s) for s in lines]

    line_height = sv.label_line_height or 1
    padding = sv.label_padding or [0, 0, 0, 0]

    ly = cy - padding[2]
    lx = cx

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

    text = _tag('text', atts, ''.join(reversed(spans)))

    # @TODO a hack to emulate 'paint-order' which wkhtmltopdf doesn't seem to support
    if atts.get('stroke'):
        atts2 = {k: v for k, v in atts.items() if not k.startswith('stroke')}
        text += _tag('text', atts2, ''.join(reversed(spans)))

    if not sv.label_background:
        return text

    width = max(xy[0] for xy in metrics) + padding[1] + padding[3]

    if anchor == 'start':
        bx = cx
    elif anchor == 'end':
        bx = cx - width
    else:
        bx = cx - width // 2

    ratts = {
        'x': bx,
        'y': cy - height,
        'width': width,
        'height': height,
        'fill': sv.label_background,
    }

    return _tag('rect', ratts) + text


def _map_font(sv):
    # @TODO: allow for more fonts and customize the mapping

    DEFAULT_FONT = 'DejaVuSans'

    if sv.label_font_weight == 'bold':
        return DEFAULT_FONT + '-Bold'
    return DEFAULT_FONT


def _fill_stroke(atts, sv, prefix=''):
    atts['fill'] = sv.get(prefix + 'fill') or 'none'

    v = sv.get(prefix + 'stroke')
    if not v:
        return

    atts['stroke'] = v

    v = sv.get(prefix + 'stroke_width')
    atts['stroke-width'] = f'{v or 1}px'

    v = sv.get(prefix + 'stroke_dasharray')
    if v:
        atts['stroke-dasharray'] = ' '.join(str(x) for x in v)

    for k in 'dashoffset', 'linecap', 'linejoin', 'miterlimit':
        v = sv.get(prefix + 'stroke_' + k)
        if v:
            atts['stroke-' + k] = v


def _tag(name, atts, content=None):
    atts = ' '.join(f'{k}="{v}"' for k, v in gws.compact(atts).items())
    if content:
        return f'<{name} {atts}>{content}</{name}>'
    return f'<{name} {atts}/>'
