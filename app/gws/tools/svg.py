import math
import base64
import shapely.ops
import shapely.geometry
from PIL import ImageFont
# import wand.image

import gws
import gws.tools.units as units
import gws.tools.xml2 as xml2
import gws.gis.extent
import gws.gis.renderview
import gws.tools.style

import gws.types as t

DEFAULT_FONT_SIZE = 10
DEFAULT_MARKER_SIZE = 10
DEFAULT_POINT_SIZE = 10

SVG_ATTRIBUTES = {
    'version': '1.1',
    'xmlns': 'http://www.w3.org/2000/svg',
}


def geometry_tags(geom: shapely.geometry.base.BaseGeometry, rv: t.MapRenderView, sv: t.StyleValues, label: str) -> t.List[t.Tag]:
    if geom.is_empty:
        return []

    trans = gws.gis.renderview.pixel_transformer(rv)
    geom = shapely.ops.transform(trans, geom)

    if not sv:
        return [_geometry(geom, {})]

    with_geometry = sv.with_geometry == gws.tools.style.StyleGeometryOption.all
    with_label = _is_label_visible(rv, sv)

    text = None

    if with_label and label:
        extra_y_offset = 0

        if sv.label_offset_y is None:
            if geom.type == 'Point':
                extra_y_offset = 12
            if geom.type == 'LineString':
                extra_y_offset = 6

        text = _label(geom, label, sv, extra_y_offset)

    marker = None
    marker_id = None

    if with_geometry and sv.marker:
        marker_id = '_M' + gws.random_string(8)
        marker = _marker(marker_id, sv)

    icon = None

    if with_geometry and sv.icon:
        res = _parse_icon_url(sv.icon, rv.dpi)
        if res:
            icon, w, h = res
            x, y, w, h = _icon_size_position(geom, sv, w, h)
            atts = {
                'x': f'{x}px',
                'y': f'{y}px',
                'width': f'{w}px',
                'height': f'{h}px',
            }
            icon += (atts,)

    body = None

    if with_geometry:
        atts = {}

        _fill_stroke_props(atts, sv)

        if marker:
            atts['marker-start'] = atts['marker-mid'] = atts['marker-end'] = f'url(#{marker_id})'

        if geom.type in ('Point', 'MultiPoint'):
            atts['r'] = (sv.point_size or DEFAULT_POINT_SIZE) // 2

        if geom.type in ('LineString', 'MultiLineString'):
            atts['fill'] = 'none'

        body = _geometry(geom, atts)

    return gws.compact([marker, body, icon, text])


def sort_by_z_index(tags: t.List[t.Tag]):
    def key(tag):
        for e in tag:
            if isinstance(e, dict) and 'z-index' in e:
                return e['z-index']
        return 0

    tags.sort(key=key)


def as_xml(tags: t.List[t.Tag]) -> str:
    return ''.join(gws.tools.xml2.as_string(tag) for tag in tags)


def as_png(tags: t.List[t.Tag], size: t.Size) -> bytes:
    sort_by_z_index(tags)
    svg = gws.tools.xml2.as_string(('svg', SVG_ATTRIBUTES, *tags))
    with wand.image.Image(blob=svg.encode('utf8'), format='svg', background='None', width=size[0], height=size[1]) as image:
        return image.make_blob('png')


def fragment_tags(fragment: t.SvgFragment, rv: t.MapRenderView) -> t.List[t.Tag]:
    """Convert an SvgFragment to a list of Tags.

    A fragment has three components:

    - a list of xml2.Tags (which are just tuples)
    - a list of points, in the map coordinate system
    - a list of named styles

    The idea is to represent client-side svg drawings (e.g. dimensions) in a resolution-independent way

    The fragment is converted as follows:

    - points are converted to pixels
    - tags' attributes are iterated. If any attribute value is an array, it's assumed to be a 'function'
    - 'class' attributes are replaced with inline styles from the .styles list

    Attribute 'functions' are

    - '' (empty function), ['', index, component] - returns `pixels[index][component]`,
        where `component` is 0 for x, 1 for y
    - 'rotate' ['rotate', index1, index2, index3] - computes a slope between `pixels[index1]` and `pixels[index2]`
        and returns a string `rotate(slope, pixels[index3])`.
    """

    trans = _pixel_transform(rv)
    pixels = [trans(*p) for p in fragment.points]
    style_map = {s.name: s.values for s in (fragment.styles or []) if s.name}

    def eval_func(v):
        if v[0] == '':
            return round(pixels[v[1]][v[2]])
        if v[0] == 'rotate':
            a = _slope(pixels[v[1]], pixels[v[2]])
            adeg = math.degrees(a)
            x, y = pixels[v[3]]
            return f'rotate({adeg:.0f}, {x:.0f}, {y:.0f})'

    def process(tag):
        for p in tag:
            if isinstance(p, (list, tuple)):
                process(p)
                continue

            if isinstance(p, dict):
                for k, v in p.items():
                    if isinstance(v, (list, tuple)):
                        p[k] = eval_func(v)
                if 'class' in p:
                    sv = style_map.get('.' + p.pop('class'))
                    if sv:
                        _fill_stroke_props(p, sv)
                        _font_props(p, sv)

    for tag in fragment.tags:
        process(tag)
    return fragment.tags


##

def _pixel_transform(rv: t.MapRenderView):
    def translate(x, y):
        x = x - rv.bounds.extent[0]
        y = rv.bounds.extent[3] - y

        return (
            units.mm2px_f((x / rv.scale) * 1000, rv.dpi),
            units.mm2px_f((y / rv.scale) * 1000, rv.dpi))

    def rotate(x, y):
        return (
            cosa * (x - ox) - sina * (y - oy) + ox,
            sina * (x - ox) + cosa * (y - oy) + oy)

    def fn(x, y):
        x, y = translate(x, y)
        if rv.rotation:
            x, y = rotate(x, y)
        return x, y

    ox, oy = translate(*gws.gis.extent.center(rv.bounds.extent))
    cosa = math.cos(math.radians(rv.rotation))
    sina = math.sin(math.radians(rv.rotation))

    return fn


def _marker(uid, sv) -> t.Tag:
    size = sv.marker_size or DEFAULT_MARKER_SIZE
    size2 = size // 2

    content = None
    atts = {}

    _fill_stroke_props(atts, sv, 'marker_')

    if sv.marker == 'circle':
        atts.update({
            'cx': size2,
            'cy': size2,
            'r': size2,
        })
        content = 'circle', atts

    if content:
        return 'marker', {
            'id': uid,
            'viewBox': f'0 0 {size} {size}',
            'refX': size2,
            'refY': size2,
            'markerUnits': 'userSpaceOnUse',
            'markerWidth': size,
            'markerHeight': size,
        }, content


def _is_label_visible(rv: t.MapRenderView, sv: t.StyleValues) -> bool:
    if sv.with_label != gws.tools.style.StyleLabelOption.all:
        return False
    if rv.scale < int(sv.get('label_min_scale', 0)):
        return False
    if rv.scale > int(sv.get('label_max_scale', 1e10)):
        return False
    return True


def _label(geom, label, sv: t.StyleValues, extra_y_offset=0) -> t.Tag:
    cx, cy = _label_position(geom, sv, extra_y_offset)
    return _text(cx, cy, label, sv)


def _label_position(geom, sv: t.StyleValues, extra_y_offset=0):
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


def _text(cx, cy, label, sv: t.StyleValues) -> t.Tag:
    # @TODO label positioning needs more work

    font_name = _map_font(sv)
    font_size = sv.label_font_size or DEFAULT_FONT_SIZE
    font = ImageFont.truetype(font=font_name, size=font_size)

    anchor = 'start'

    if sv.label_align == 'right':
        anchor = 'end'
    elif sv.label_align == 'center':
        anchor = 'middle'

    atts = {'text-anchor': anchor}

    _font_props(atts, sv, 'label_')
    _fill_stroke_props(atts, sv, 'label_')

    lines = label.split('\n')
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
        spans.append(xml2.tag('tspan', {'x': lx, 'y': ly}, s))
        ly -= (em_height + line_height)

    tags = [xml2.tag('text', atts, *reversed(spans))]

    # @TODO a hack to emulate 'paint-order' which wkhtmltopdf doesn't seem to support
    # place a copy without the stroke above above the text
    if atts.get('stroke'):
        no_stroke_atts = {k: v for k, v in atts.items() if not k.startswith('stroke')}
        tags.append(xml2.tag('text', no_stroke_atts, *reversed(spans)))

    if sv.label_background:
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

        tags.insert(0, xml2.tag('rect', ratts))

    # a hack to move labels forward: emit a (non-supported) z-index attribute
    # and sort elements by it later on

    return xml2.tag('g', {'z-index': 100}, *tags)


_PFX_SVF_UTF8 = 'data:image/svg+xml;utf8,'
_PFX_SVF_BASE64 = 'data:image/svg+xml;base64,'


def _parse_icon_url(url, dpi):
    src = None

    if url.startswith(_PFX_SVF_UTF8):
        src = url[len(_PFX_SVF_UTF8):]
    elif url.startswith(_PFX_SVF_BASE64):
        src = base64.standard_b64decode(url[len(_PFX_SVF_BASE64):]).decode('utf8')

    if not src:
        return

    try:
        svg = xml2.from_string(src)
    except xml2.Error as e:
        gws.log.error(f'error parsing icon url: {e}')
        return

    w = svg.attr('width')
    h = svg.attr('height')

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

    sub = gws.compact(_clean(c) for c in svg.children)

    return xml2.tag('svg', *(c.as_tag() for c in sub)), w, h


_ALLOWED_TAGS = {
    'circle',
    'clippath',
    'defs',
    'ellipse',
    'g',
    'hatch',
    'hatchpath',
    'line',
    'lineargradient',
    'marker',
    'mask',
    'mesh',
    'meshgradient',
    'meshpatch',
    'meshrow',
    'mpath',
    'path',
    'pattern',
    'polygon',
    'polyline',
    'radialgradient',
    'rect',
    'solidcolor',
    'symbol',
    'text',
    'textpath',
    'title',
    'tspan',
    'use',
}


def _clean(el: xml2.Element) -> t.Optional[xml2.Element]:
    # @TODO: security! since icons are arbitrary content, check for valid tag names and values

    if el.name not in _ALLOWED_TAGS:
        return

    el.attributes = gws.compact(_clean_attr(a) for a in el.attributes)
    el.children = gws.compact(_clean(c) for c in el.children)

    return el


def _clean_attr(a: xml2.Attribute):
    if a.value.startswith(('http:', 'https:', 'data:')):
        return
    return a


def _icon_size_position(geom, sv, width, height):
    # @TODO style options for icon positioning
    c = geom.centroid
    return (
        int(c.x - width / 2),
        int(c.y - height / 2),
        width,
        height)


def _get_points(geom):
    gt = geom.type

    if gt in ('Point', 'LineString', 'LinearRing'):
        return geom.coords
    if gt in ('Polygon', 'LineString', 'LinearRing'):
        return geom.exterior.coords
    if gt.startswith('Multi'):
        return [p for g in geom.geoms for p in _get_points(g)]


def _fill_stroke_props(atts, sv, prefix=''):
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


def _font_props(atts, sv, prefix=''):
    font_name = _map_font(sv, prefix)
    font_size = sv.get(prefix + 'font_size') or DEFAULT_FONT_SIZE

    atts.update(gws.compact({
        'font-family': font_name.split('-')[0],
        'font-size': f'{font_size}px',
        'font-weight': sv.get(prefix + 'font_weight'),
        'font-style': sv.get(prefix + 'font_style'),
    }))


def _map_font(sv, prefix=''):
    # @TODO: allow for more fonts and customize the mapping

    DEFAULT_FONT = 'DejaVuSans'
    w = sv.get(prefix + 'font_weight')
    if w == 'bold':
        return DEFAULT_FONT + '-Bold'
    return DEFAULT_FONT


def _geometry(geom: shapely.geometry.base.BaseGeometry, atts: dict) -> t.Tag:
    def _xy(xy):
        return str(xy[0]) + ' ' + str(xy[1])

    def _lpath(coords):
        ps = []
        cs = iter(coords)
        for c in cs:
            ps.append(f'M {_xy(c)}')
            break
        for c in cs:
            ps.append(f'L {_xy(c)}')
        return ' '.join(ps)

    ty = geom.type.lower()

    if ty == 'point':
        g = t.cast(shapely.geometry.Point, geom)
        return xml2.tag('circle', {'cx': g.x, 'cy': g.y}, atts)

    if ty == 'linestring':
        g = t.cast(shapely.geometry.LineString, geom)
        d = _lpath(g.coords)
        return xml2.tag('path', {'d': d}, atts)

    if ty == 'polygon':
        g = t.cast(shapely.geometry.Polygon, geom)
        d = ' '.join(_lpath(interior.coords) + ' z' for interior in g.interiors)
        d = _lpath(g.exterior.coords) + ' z ' + d
        return xml2.tag('path', {'fill-rule': 'evenodd', 'd': d.strip()}, atts)

    if ty in ('multipolygon', 'multipoint', 'mutlilinestring'):
        g = t.cast(shapely.geometry.base.BaseMultipartGeometry, geom)
        return xml2.tag('g', *[_geometry(p, atts) for p in g.geoms])

    raise ValueError(f'unknown type {geom.type!r}')


def _slope(a: t.Point, b: t.Point) -> float:
    # slope between two points
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if dx == 0:
        dx = 0.01;

    return math.atan(dy / dx)


def _rotate(p: t.Point, o: t.Point, a: float) -> t.Point:
    # rotate P(oint) about O(rigin) by A(ngle)
    return (
        o[0] + (p[0] - o[0]) * math.cos(a) - (p[1] - o[1]) * math.sin(a),
        o[1] + (p[0] - o[0]) * math.sin(a) + (p[1] - o[1]) * math.cos(a),
    )
