import base64
import math
import shapely.geometry
import shapely.ops

import gws
import gws.lib.font
import gws.gis.render
import gws.gis.shape
import gws.lib.units as units
import gws.lib.xml3 as xml3
import gws.types as t

DEFAULT_FONT_SIZE = 10
DEFAULT_MARKER_SIZE = 10
DEFAULT_POINT_SIZE = 10


def shape_to_fragment(shape: gws.IShape, view: gws.MapView, style: gws.IStyle = None, label: str = None) -> t.List[gws.XmlElement]:
    if not shape:
        return []

    geom = t.cast(gws.gis.shape.Shape, shape).geom
    if geom.is_empty:
        return []

    trans = gws.gis.render.map_view_transformer(view)
    geom = shapely.ops.transform(trans, geom)

    if not style:
        return [_geometry(geom)]

    sv = style.values
    with_geometry = sv.with_geometry == 'all'
    with_label = label and _is_label_visible(view, sv)
    gt = _geom_type(geom)

    text = None
    if with_label:
        extra_y_offset = 0
        if sv.label_offset_y is None:
            if gt == _TYPE_POINT:
                extra_y_offset = 12
            if gt == _TYPE_LINESTRING:
                extra_y_offset = 6
        text = _label(geom, label, sv, extra_y_offset)

    marker = None
    marker_id = None
    if with_geometry and sv.marker:
        marker_id = '_M' + gws.random_string(8)
        marker = _marker(marker_id, sv)

    icon = None
    atts: dict = {}
    if with_geometry and sv.icon:
        res = _parse_icon(sv.icon, view.dpi)
        if res:
            el, w, h = res
            x, y, w, h = _icon_size_position(geom, sv, w, h)
            atts = {
                'x': f'{int(x)}',
                'y': f'{int(y)}',
                'width': f'{int(w)}',
                'height': f'{int(h)}',
            }
            icon = xml3.element(
                name=el.name,
                attributes=gws.merge(el.attributes, atts),
                children=el.children)

    body = None
    if with_geometry:
        _add_paint_atts(atts, sv)
        if marker:
            atts['marker-start'] = atts['marker-mid'] = atts['marker-end'] = f'url(#{marker_id})'
        if gt == _TYPE_POINT or gt == _TYPE_MULTIPOINT:
            atts['r'] = (sv.point_size or DEFAULT_POINT_SIZE) // 2
        if gt == _TYPE_LINESTRING or gt == _TYPE_MUTLILINESTRING:
            atts['fill'] = 'none'
        body = _geometry(geom, atts)

    return gws.compact([marker, body, icon, text])


# ----------------------------------------------------------------------------------------------------------------------
# soup

def soup_to_fragment(view: gws.MapView, points: t.List[gws.Point], tags: t.List[t.Any]) -> t.List[gws.XmlElement]:
    """Convert an svg "soup" to a list of XmlElements.

    A soup has two components:

    - a list of points, in the map coordinate system
    - a list of tuples suitable for `xml3.tag` input (tag-name, {atts}, child1, child2....)

    The idea is to represent client-side svg drawings (e.g. dimensions) in a resolution-independent way

    First, points are converted to pixels using the view's transform. Then, each tag's attributes are iterated.
    If any attribute value is an array, it's assumed to be a 'function'.
    The first element is a function name, the rest are arguments.
    Attribute 'functions' are

    - ['x', n] - returns points[n][0]
    - ['y', n] - returns points[n][1]
    - ['r', p1, p2, r] - computes a slope between points[p1] points[p2] and returns a string
        `rotate(slope, points[r].x, points[r].y)`

    """

    trans = gws.gis.render.map_view_transformer(view)
    px = [trans(*p) for p in points]

    def eval_func(v):
        if v[0] == 'x':
            return round(px[v[1]][0])
        if v[0] == 'y':
            return round(px[v[1]][1])
        if v[0] == 'r':
            a = _slope(px[v[1]], px[v[2]])
            adeg = math.degrees(a)
            x, y = px[v[3]]
            return f'rotate({adeg:.0f}, {x:.0f}, {y:.0f})'

    def convert(tag):
        for arg in tag:
            if isinstance(arg, (list, tuple)):
                convert(arg)
            elif isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, (list, tuple)):
                        arg[k] = eval_func(v)

    els = []

    for tag in tags:
        convert(tag)
        els.append(xml3.tag(*tag))

    return els


def _slope(a: gws.Point, b: gws.Point) -> float:
    # slope between two points
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if dx == 0:
        dx = 0.01

    return math.atan(dy / dx)


# ----------------------------------------------------------------------------------------------------------------------
# geometry

_TYPE_POINT = 1
_TYPE_LINESTRING = 2
_TYPE_POLYGON = 3
_TYPE_LINEARRING = 4
_TYPE_MULTI = 0x10
_TYPE_MULTIPOINT = _TYPE_MULTI | _TYPE_POINT
_TYPE_MUTLILINESTRING = _TYPE_MULTI | _TYPE_LINESTRING
_TYPE_MULTIPOLYGON = _TYPE_MULTI | _TYPE_POLYGON


def _geometry(geom: shapely.geometry.base.BaseGeometry, atts: dict = None) -> gws.XmlElement:
    def _xy(xy):
        x, y = xy
        return f'{x} {y}'

    def _lpath(coords):
        ps = []
        cs = iter(coords)
        for c in cs:
            ps.append(f'M {_xy(c)}')
            break
        for c in cs:
            ps.append(f'L {_xy(c)}')
        return ' '.join(ps)

    gt = _geom_type(geom)

    if gt == _TYPE_POINT:
        g = t.cast(shapely.geometry.Point, geom)
        return xml3.tag('circle', {'cx': int(g.x), 'cy': int(g.y)}, atts)

    if gt == _TYPE_LINESTRING:
        g = t.cast(shapely.geometry.LineString, geom)
        d = _lpath(g.coords)
        return xml3.tag('path', {'d': d}, atts)

    if gt == _TYPE_POLYGON:
        g = t.cast(shapely.geometry.Polygon, geom)
        d = ' '.join(_lpath(interior.coords) + ' z' for interior in g.interiors)
        d = _lpath(g.exterior.coords) + ' z ' + d
        return xml3.tag('path', {'fill-rule': 'evenodd', 'd': d.strip()}, atts)

    if gt > _TYPE_MULTI:
        g = t.cast(shapely.geometry.base.BaseMultipartGeometry, geom)
        return xml3.tag('g', *[_geometry(p, atts) for p in g.geoms])


def _enum_points(geom):
    gt = _geom_type(geom)

    if gt == _TYPE_POINT or gt == _TYPE_LINESTRING or gt == _TYPE_LINEARRING:
        return geom.coords
    if gt == _TYPE_POLYGON:
        return geom.exterior.coords
    if gt > _TYPE_MULTI:
        return [p for g in geom.geoms for p in _enum_points(g)]


_geom_types = {
    'point': _TYPE_POINT,
    'linestring': _TYPE_LINESTRING,
    'polygon': _TYPE_POLYGON,
    'linearring': _TYPE_LINEARRING,
    'multipoint': _TYPE_MULTIPOINT,
    'mutlilinestring': _TYPE_MUTLILINESTRING,
    'multipolygon': _TYPE_MULTIPOLYGON,
}


def _geom_type(geom):
    try:
        return _geom_types[geom.type.lower()]
    except KeyError:
        raise ValueError(f'unknown type {geom.type!r}')


# ----------------------------------------------------------------------------------------------------------------------
# marker

# @TODO only type=circle is implemented

def _marker(uid, sv: gws.StyleValues) -> gws.XmlElement:
    size = sv.marker_size or DEFAULT_MARKER_SIZE
    size2 = size // 2

    content = None
    atts: dict = {}

    _add_paint_atts(atts, sv, 'marker_')

    if sv.marker == 'circle':
        atts.update({
            'cx': size2,
            'cy': size2,
            'r': size2,
        })
        content = 'circle', atts

    if content:
        return xml3.tag('marker', {
            'id': uid,
            'viewBox': f'0 0 {size} {size}',
            'refX': size2,
            'refY': size2,
            'markerUnits': 'userSpaceOnUse',
            'markerWidth': size,
            'markerHeight': size,
        }, content)


# ----------------------------------------------------------------------------------------------------------------------
# labels

# @TODO label positioning needs more work

def _is_label_visible(view: gws.MapView, sv: gws.StyleValues) -> bool:
    if sv.with_label != 'all':
        return False
    if view.scale < int(sv.get('label_min_scale', 0)):
        return False
    if view.scale > int(sv.get('label_max_scale', 1e10)):
        return False
    return True


def _label(geom, label: str, sv: gws.StyleValues, extra_y_offset=0) -> gws.XmlElement:
    xy = _label_position(geom, sv, extra_y_offset)
    return _label_text(xy[0], xy[1], label, sv)


def _label_position(geom, sv: gws.StyleValues, extra_y_offset=0) -> gws.Point:
    if sv.label_placement == 'start':
        x, y = _enum_points(geom)[0]
    elif sv.label_placement == 'end':
        x, y = _enum_points(geom)[-1]
    else:
        c = geom.centroid
        x, y = c.x, c.y
    return (
        round(x) + (sv.label_offset_x or 0),
        round(y) + extra_y_offset + (sv.label_font_size >> 1) + (sv.label_offset_y or 0)
    )


def _label_text(cx, cy, label, sv: gws.StyleValues) -> gws.XmlElement:
    font_name = _font_name(sv)
    font_size = sv.label_font_size or DEFAULT_FONT_SIZE
    font = gws.lib.font.from_name(font_name, font_size)

    anchor = 'start'

    if sv.label_align == 'right':
        anchor = 'end'
    elif sv.label_align == 'center':
        anchor = 'middle'

    atts = {'text-anchor': anchor}

    _add_font_atts(atts, sv, 'label_')
    _add_paint_atts(atts, sv, 'label_')

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
        spans.append(['tspan', {'x': lx, 'y': ly}, s])
        ly -= (em_height + line_height)

    tags = []

    tags.append(('text', atts, *reversed(spans)))

    # @TODO a hack to emulate 'paint-order' which wkhtmltopdf doesn't seem to support
    # place a copy without the stroke above the text
    if atts.get('stroke'):
        no_stroke_atts = {k: v for k, v in atts.items() if not k.startswith('stroke')}
        tags.append(('text', no_stroke_atts, *reversed(spans)))

    # @TODO label backgrounds don't really work
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

        tags.insert(0, ('rect', ratts))

    # a hack to move labels forward: emit a (non-supported) z-index attribute
    # and sort elements by it later on (see `fragment_to_element`)

    return xml3.tag('g', {'z-index': 100}, *tags)


# ----------------------------------------------------------------------------------------------------------------------
# icons

# @TODO options for icon positioning


def _parse_icon(icon, dpi):
    # see lib.style.icon

    svg = None
    if gws.is_data_object(icon):
        svg = icon.svg
    if not svg:
        return

    w = xml3.attr(svg, 'width')
    h = xml3.attr(svg, 'height')

    if not w or not h:
        gws.log.error(f'xml_icon: width and height required')
        return

    try:
        w, wu = units.parse(w, default=units.PX)
        h, hu = units.parse(h, default=units.PX)
    except ValueError:
        gws.log.error(f'xml_icon: invalid units: {w!r} {h!r}')
        return

    if wu == 'mm':
        w = units.mm_to_px(w, dpi)
    if hu == 'mm':
        h = units.mm_to_px(h, dpi)

    return svg, w, h


def _icon_size_position(geom, sv, width, height):
    c = geom.centroid
    return (
        int(c.x - width / 2),
        int(c.y - height / 2),
        width,
        height)


# ----------------------------------------------------------------------------------------------------------------------
# fonts

# @TODO: allow for more fonts and customize the mapping


_DEFAULT_FONT = 'DejaVuSans'


def _add_font_atts(atts, sv, prefix=''):
    font_name = _font_name(sv, prefix)
    font_size = sv.get(prefix + 'font_size') or DEFAULT_FONT_SIZE

    atts.update(gws.compact({
        'font-family': font_name.split('-')[0],
        'font-size': f'{font_size}px',
        'font-weight': sv.get(prefix + 'font_weight'),
        'font-style': sv.get(prefix + 'font_style'),
    }))


def _font_name(sv, prefix=''):
    w = sv.get(prefix + 'font_weight')
    if w == 'bold':
        return _DEFAULT_FONT + '-Bold'
    return _DEFAULT_FONT


# ----------------------------------------------------------------------------------------------------------------------
# paint

def _add_paint_atts(atts, sv, prefix=''):
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
