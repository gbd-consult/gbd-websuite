"""Build SVG fragments from shapes and drawing soups."""

from typing import Optional, cast

import base64
import math
import shapely
import shapely.geometry
import shapely.ops

import gws
import gws.lib.extent
import gws.lib.font
import gws.lib.shape
import gws.lib.uom
import gws.lib.xmlx as xmlx

from . import element

DEFAULT_FONT_SIZE = 10
DEFAULT_MARKER_SIZE = 10
DEFAULT_POINT_SIZE = 10

MAX_SOUP_POINTS = 5000
MAX_SOUP_TAGS = 5000


def shape_to_fragment(shape: gws.Shape, view: gws.MapView, label: str = None, style: gws.Style = None) -> list[gws.XmlElement]:
    """Convert a shape to an SVG fragment.

    The shape is transformed to pixel coordinates using the view. Without a style, only the geometry is drawn.
    With a style, the fragment can contain a marker definition, the geometry, an icon and a label,
    depending on the style values and the view scale.

    Args:
        shape: Shape to draw.
        view: Map view, which defines the pixel transformation, scale and dpi.
        label: Label text, can contain newlines.
        style: Style to apply.

    Returns:
        A list of SVG elements, empty if the shape is missing or empty.

    Raises:
        ``gws.Error``: If the geometry type is not supported.
    """

    if not shape:
        return []

    geom = cast(gws.lib.shape.Shape, shape).geom
    if geom.is_empty:
        return []

    trans = _map_view_transformer(view)
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
                extra_y_offset = (sv.label_font_size or DEFAULT_FONT_SIZE) * 2
            if gt == _TYPE_LINESTRING:
                extra_y_offset = 6
        text = _label(geom, label, sv, extra_y_offset)

    marker = None
    marker_id = None

    if with_geometry and sv.marker:
        marker_id = '_M' + gws.u.random_string(8)
        marker = _marker(marker_id, sv)

    atts: dict = {}

    icon = None

    if with_geometry and sv.icon is not None:
        res = _parse_icon(sv.icon, view.dpi)
        if res:
            icon_el, w, h = res
            x, y, w, h = _icon_size_and_position(geom, sv, w, h)
            atts = {
                'x': f'{int(x)}',
                'y': f'{int(y)}',
                'width': f'{int(w)}',
                'height': f'{int(h)}',
            }
            icon = xmlx.tag(
                icon_el.name,
                gws.u.merge(icon_el.attrib, atts),
                *icon_el.children()
            )

    body = None

    if with_geometry:
        _add_paint_atts(atts, sv)
        if marker:
            atts['marker-start'] = atts['marker-mid'] = atts['marker-end'] = f'url(#{marker_id})'
        if gt in {_TYPE_POINT, _TYPE_MULTIPOINT}:
            atts['r'] = (sv.point_size or DEFAULT_POINT_SIZE) // 2
        if gt in {_TYPE_LINESTRING, _TYPE_MULTILINESTRING}:
            atts['fill'] = 'none'
        body = _geometry(geom, atts)

    return gws.u.compact([marker, body, icon, text])


def soup_to_fragment(view: gws.MapView, points: list[gws.Point], tags: list) -> list[gws.XmlElement]:
    """Convert an SVG "soup" to an SVG fragment.

    A soup represents client-side SVG drawings (e.g. dimensions) in a resolution-independent way.
    It has two components:

    - a list of points, in the map coordinate system,
    - a list of tuples ``(tag-name, {atts}, child1, child2, ...)``, where children are tuples of the same form or strings.

    First, points are converted to pixels using the view's transform. Then, the attributes of each tag are evaluated.
    If an attribute value is a list, it is a function call: the first element is a function name,
    the rest are arguments. The functions are:

    - ``['x', n]``: returns the x pixel coordinate of ``points[n]``,
    - ``['y', n]``: returns the y pixel coordinate of ``points[n]``,
    - ``['r', p1, p2, r]``: computes the slope between ``points[p1]`` and ``points[p2]`` and returns
      ``rotate(slope, x, y)`` with the coordinates of ``points[r]``.

    The result is normalized, so unsafe tags and attributes are removed.

    Args:
        view: Map view, which defines the pixel transformation.
        points: Points in map coordinates.
        tags: Tag tuples.

    Returns:
        A list of SVG elements.

    Raises:
        ``gws.Error``: If the soup is too large or invalid, or uses an unknown function.
    """

    if len(points) > MAX_SOUP_POINTS:
        raise gws.Error(f'too many soup points: {len(points)}')
    if len(tags) > MAX_SOUP_TAGS:
        raise gws.Error(f'too many soup tags: {len(tags)}')

    trans = _map_view_transformer(view)

    try:
        px = [trans(*p) for p in points]
    except Exception as exc:
        raise gws.Error('invalid soup') from exc

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
        raise gws.Error(f'unknown soup function: {v[0]!r}')

    def eval_funcs(tag):
        res = []
        for arg in tag:
            if isinstance(arg, dict):
                d = {}
                for k, v in arg.items():
                    d[k] = eval_func(v) if isinstance(v, (list, tuple)) else v
                res.append(d)
            elif isinstance(arg, (list, tuple)):
                res.append(eval_funcs(arg))
            else:
                res.append(arg)
        return res

    def soup_tag(ls):
        args = [soup_tag(a) if isinstance(a, (list, tuple)) else a for a in ls[1:]]
        return xmlx.tag(ls[0], *args)

    els = []

    try:
        for tag in tags:
            els.append(soup_tag(eval_funcs(tag)))
    except Exception as exc:
        raise gws.Error('invalid soup') from exc

    return element.normalize_fragment(els)


# ----------------------------------------------------------------------------------------------------------------------
# transform

def _map_view_transformer(view: gws.MapView):
    """Create a transformer from map coordinates to pixel coordinates of a view.

    Pixel coordinates are integers, relative to the top left corner of the view
    at the view's scale and DPI. For a rotated view, points are rotated around the view center.

    Args:
        view: Map view.

    Returns:
        A function ``f(x, y) -> (px, py)``.
    """

    # @TODO cache the transformer

    def translate(x, y):
        x = x - ext[0]
        y = ext[3] - y
        return x * m2px, y * m2px

    def translate_int(x, y):
        x, y = translate(x, y)
        return int(x), int(y)

    def rotate(x, y):
        return (
            cosa * (x - ox) - sina * (y - oy) + ox,
            sina * (x - ox) + cosa * (y - oy) + oy)

    def translate_rotate_int(x, y):
        x, y = translate(x, y)
        x, y = rotate(x, y)
        return int(x), int(y)

    m2px = 1000.0 * gws.lib.uom.mm_to_px(1 / view.scale, view.dpi)

    ext = view.bounds.extent

    if not view.rotation:
        return translate_int

    ox, oy = translate(*gws.lib.extent.center(ext))
    cosa = math.cos(math.radians(view.rotation))
    sina = math.sin(math.radians(view.rotation))

    return translate_rotate_int


# ----------------------------------------------------------------------------------------------------------------------
# geometry

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
        g = cast(shapely.geometry.Point, geom)
        return xmlx.tag('circle', {'cx': int(g.x), 'cy': int(g.y)}, atts)

    if gt == _TYPE_LINESTRING:
        g = cast(shapely.geometry.LineString, geom)
        d = _lpath(g.coords)
        return xmlx.tag('path', {'d': d}, atts)

    if gt == _TYPE_POLYGON:
        g = cast(shapely.geometry.Polygon, geom)
        d = ' '.join(_lpath(interior.coords) + ' z' for interior in g.interiors)
        d = _lpath(g.exterior.coords) + ' z ' + d
        return xmlx.tag('path', {'fill-rule': 'evenodd', 'd': d.strip()}, atts)

    if gt >= _TYPE_MULTIPOINT:
        g = cast(shapely.geometry.base.BaseMultipartGeometry, geom)
        return xmlx.tag('g', *[_geometry(p, atts) for p in g.geoms])


def _enum_points(geom):
    gt = _geom_type(geom)

    if gt in {_TYPE_POINT, _TYPE_LINESTRING, _TYPE_LINEARRING}:
        return geom.coords
    if gt == _TYPE_POLYGON:
        return geom.exterior.coords
    if gt >= _TYPE_MULTIPOINT:
        return [p for g in geom.geoms for p in _enum_points(g)]


# https://shapely.readthedocs.io/en/stable/reference/shapely.get_type_id.html

_TYPE_POINT = 0
_TYPE_LINESTRING = 1
_TYPE_LINEARRING = 2
_TYPE_POLYGON = 3
_TYPE_MULTIPOINT = 4
_TYPE_MULTILINESTRING = 5
_TYPE_MULTIPOLYGON = 6
_TYPE_GEOMETRYCOLLECTION = 7


def _geom_type(geom):
    p = shapely.get_type_id(geom)
    if _TYPE_POINT <= p <= _TYPE_MULTIPOLYGON:
        return p
    raise gws.Error(f'unsupported geometry type {geom.type!r}')


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
        content = xmlx.tag('circle', atts)

    if content:
        return xmlx.tag('marker', {
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
    _, em_height = _font_size(font, 'MMM')
    metrics = [_font_size(font, s) for s in lines]

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
        spans.append(xmlx.tag('tspan', {'x': lx, 'y': ly}, s))
        ly -= (em_height + line_height)

    tags = []

    tags.append(xmlx.tag('text', atts, *reversed(spans)))

    # @TODO a hack to emulate 'paint-order' which wkhtmltopdf doesn't seem to support
    # place a copy without the stroke above the text
    if atts.get('stroke'):
        no_stroke_atts = {k: v for k, v in atts.items() if not k.startswith('stroke')}
        tags.append(xmlx.tag('text', no_stroke_atts, *reversed(spans)))

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

        tags.insert(0, xmlx.tag('rect', ratts))

    # a hack to move labels forward: emit a (non-supported) z-index attribute
    # and sort elements by it later on (see `fragment_to_element`)

    return xmlx.tag('g', {'z-index': 100}, *tags)


# ----------------------------------------------------------------------------------------------------------------------
# icons

# @TODO options for icon positioning


def _parse_icon(svg: gws.XmlElement, dpi) -> Optional[tuple[gws.XmlElement, float, float]]:
    # see lib.style.icon

    if not isinstance(svg, gws.XmlElement):
        return

    w = svg.get('width')
    h = svg.get('height')

    if not w or not h:
        gws.log.error(f'xml_icon: width and height required')
        return

    try:
        w, wu = gws.lib.uom.parse(w, gws.Uom.px)
        h, hu = gws.lib.uom.parse(h, gws.Uom.px)
    except ValueError:
        gws.log.error(f'xml_icon: invalid units: {w!r} {h!r}')
        return

    if wu == gws.Uom.mm:
        w = gws.lib.uom.mm_to_px(w, dpi)
    if hu == gws.Uom.mm:
        h = gws.lib.uom.mm_to_px(h, dpi)

    return svg, w, h


def _icon_size_and_position(geom, sv, width, height) -> tuple[int, int, int, int]:
    c = geom.centroid
    return (
        int(c.x - width / 2),
        int(c.y - height / 2),
        int(width),
        int(height))


# ----------------------------------------------------------------------------------------------------------------------
# fonts

# @TODO: allow for more fonts and customize the mapping


_DEFAULT_FONT = 'DejaVuSans'


def _add_font_atts(atts, sv, prefix=''):
    font_name = _font_name(sv, prefix)
    font_size = sv.get(prefix + 'font_size') or DEFAULT_FONT_SIZE

    atts.update(gws.u.compact({
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


def _font_size(font, text):
    bb = font.getbbox(text)
    return bb[2] - bb[0], bb[3] - bb[1]


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


# ----------------------------------------------------------------------------------------------------------------------
# misc

def _slope(a: gws.Point, b: gws.Point) -> float:
    # slope between two points
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if dx == 0:
        dx = 0.01

    return math.atan(dy / dx)
