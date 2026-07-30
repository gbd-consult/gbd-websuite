# normalizer

from typing import Optional, Dict, Pattern

import re

import gws
import gws.lib.xmlx as xmlx
import gws.lib.mime
import gws.lib.image

_SVG_TAG_ATTS = {
    'xmlns': 'http://www.w3.org/2000/svg',
}


def fragment_to_element(fragment: list[gws.XmlElement], atts: dict = None) -> gws.XmlElement:
    """Convert an SVG fragment to an SVG element."""

    fr = sorted(fragment, key=lambda el: el.attrib.get('z-index', 0))
    return xmlx.tag('svg', _SVG_TAG_ATTS, atts, *fr)


def fragment_to_image(fragment: list[gws.XmlElement], size: gws.Size, mime=gws.lib.mime.PNG) -> gws.lib.image.Image:
    """Convert an SVG fragment to a raster image."""

    el = fragment_to_element(fragment)
    return gws.lib.image.from_svg(el.to_string(), size, mime)


def normalize_element(el: gws.XmlElement) -> gws.XmlElement:
    """Remove unsafe stuff from an SVG element and normalize tag and attribute names."""

    children = gws.u.compact(_normalize(c) for c in el)
    return xmlx.tag('svg', _SVG_TAG_ATTS, _normalize_atts(el.attrib), *children)


def normalize_fragment(fragment: list[gws.XmlElement]) -> list[gws.XmlElement]:
    """Remove unsafe stuff from an SVG fragment and normalize tag and attribute names."""

    els = [_normalize(el) for el in fragment]
    return [el for el in els if el is not None]


##

_ALLOWED_TAGS = {
    'circle',
    'clipPath',
    'defs',
    'ellipse',
    'g',
    'hatch',
    'hatchpath',
    'line',
    'linearGradient',
    'marker',
    'mask',
    'mesh',
    'meshgradient',
    'meshpatch',
    'meshrow',
    'path',
    'pattern',
    'polygon',
    'polyline',
    'radialGradient',
    'rect',
    'solidcolor',
    'symbol',
    'text',
    'title',
    'tspan',
}

# tags whose text content is rendered

_TEXT_TAGS = {
    'text',
    'title',
    'tspan',
}

_CANONICAL_TAGS = {s.lower(): s for s in _ALLOWED_TAGS}

# Regex patterns for attribute validation
_RE_COLOR = r'^(#[0-9A-Fa-f]{3,8}|(rgb|rgba|hsl|hsla)\([\d%,.\s]+\)|aliceblue|antiquewhite|aqua|aquamarine|azure|beige|bisque|black|blanchedalmond|blue|blueviolet|brown|burlywood|cadetblue|chartreuse|chocolate|coral|cornflowerblue|cornsilk|crimson|cyan|darkblue|darkcyan|darkgoldenrod|darkgray|darkgreen|darkgrey|darkkhaki|darkmagenta|darkolivegreen|darkorange|darkorchid|darkred|darksalmon|darkseagreen|darkslateblue|darkslategray|darkslategrey|darkturquoise|darkviolet|deeppink|deepskyblue|dimgray|dimgrey|dodgerblue|firebrick|floralwhite|forestgreen|fuchsia|gainsboro|ghostwhite|gold|goldenrod|gray|green|greenyellow|grey|honeydew|hotpink|indianred|indigo|ivory|khaki|lavender|lavenderblush|lawngreen|lemonchiffon|lightblue|lightcoral|lightcyan|lightgoldenrodyellow|lightgray|lightgreen|lightgrey|lightpink|lightsalmon|lightseagreen|lightskyblue|lightslategray|lightslategrey|lightsteelblue|lightyellow|lime|limegreen|linen|magenta|maroon|mediumaquamarine|mediumblue|mediumorchid|mediumpurple|mediumseagreen|mediumslateblue|mediumspringgreen|mediumturquoise|mediumvioletred|midnightblue|mintcream|mistyrose|moccasin|navajowhite|navy|oldlace|olive|olivedrab|orange|orangered|orchid|palegoldenrod|palegreen|paleturquoise|palevioletred|papayawhip|peachpuff|peru|pink|plum|powderblue|purple|red|rosybrown|royalblue|saddlebrown|salmon|sandybrown|seagreen|seashell|sienna|silver|skyblue|slateblue|slategray|slategrey|snow|springgreen|steelblue|tan|teal|thistle|tomato|turquoise|violet|wheat|white|whitesmoke|yellow|yellowgreen|transparent|currentColor)$'
_RE_NUMBER = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?(px|em|ex|pt|pc|cm|mm|in|%)?$'
_RE_OPACITY = r'^(0(\.\d+)?|1(\.0+)?)$'
_RE_PATH = r'^[mMlLhHvVcCsSqQtTaAzZ0-9\s,.-]+$'
_RE_TRANSFORM = r'^(matrix|translate|scale|rotate|skewX|skewY)\([\d\s,.-]+\)( (matrix|translate|scale|rotate|skewX|skewY)\([\d\s,.-]+\))*$'
_RE_VIEWBOX = r'^[-+]?\d+(\.\d+)?([\s,]+[-+]?\d+(\.\d+)?){3}$'
_RE_TEXT = r'^[^<>]*$'
_RE_FONT_FAMILY = r'^[^<>"\']*$'
_RE_NAME = r'^[A-Za-z0-9_-]+$'
_RE_NAME_LIST = r'^[A-Za-z0-9_ -]*$'
_RE_NUMBER_LIST = r'^[\d\s,.-]+$'

# Dictionary of allowed attributes with their validation patterns
_ALLOWED_ATTRIBUTES: Dict[str, Pattern] = {
    'alignment-baseline': re.compile(_RE_TEXT),
    'baseline-shift': re.compile(_RE_TEXT),
    'class': re.compile(_RE_NAME_LIST),
    'clip': re.compile(_RE_TEXT),
    'clip-path': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'clip-rule': re.compile(r'^(nonzero|evenodd)$'),
    'color': re.compile(_RE_COLOR),
    'color-interpolation': re.compile(r'^(auto|sRGB|linearRGB)$'),
    'color-interpolation-filters': re.compile(r'^(auto|sRGB|linearRGB)$'),
    'color-profile': re.compile(_RE_TEXT),
    'color-rendering': re.compile(r'^(auto|optimizeSpeed|optimizeQuality)$'),
    'cursor': re.compile(_RE_TEXT),
    'cx': re.compile(_RE_NUMBER),
    'cy': re.compile(_RE_NUMBER),
    'd': re.compile(_RE_PATH),
    'direction': re.compile(r'^(ltr|rtl)$'),
    'display': re.compile(r'^(inline|block|list-item|run-in|compact|marker|table|inline-table|table-row-group|table-header-group|table-footer-group|table-row|table-column-group|table-column|table-cell|table-caption|none)$'),
    'dominant-baseline': re.compile(_RE_TEXT),
    'dx': re.compile(_RE_NUMBER),
    'dy': re.compile(_RE_NUMBER),
    'enable-background': re.compile(_RE_TEXT),
    'fill': re.compile(_RE_COLOR),
    'fill-opacity': re.compile(_RE_OPACITY),
    'fill-rule': re.compile(r'^(nonzero|evenodd)$'),
    'filter': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'flood-color': re.compile(_RE_COLOR),
    'flood-opacity': re.compile(_RE_OPACITY),
    'font-family': re.compile(_RE_FONT_FAMILY),
    'font-size': re.compile(_RE_NUMBER),
    'font-size-adjust': re.compile(_RE_NUMBER),
    'font-stretch': re.compile(r'^(normal|wider|narrower|ultra-condensed|extra-condensed|condensed|semi-condensed|semi-expanded|expanded|extra-expanded|ultra-expanded)$'),
    'font-style': re.compile(r'^(normal|italic|oblique)$'),
    'font-variant': re.compile(r'^(normal|small-caps)$'),
    'font-weight': re.compile(r'^(normal|bold|bolder|lighter|100|200|300|400|500|600|700|800|900)$'),
    'glyph-orientation-horizontal': re.compile(_RE_NUMBER),
    'glyph-orientation-vertical': re.compile(_RE_NUMBER),
    'id': re.compile(_RE_NAME),
    'image-rendering': re.compile(r'^(auto|optimizeSpeed|optimizeQuality)$'),
    'kerning': re.compile(_RE_TEXT),
    'letter-spacing': re.compile(_RE_NUMBER),
    'lighting-color': re.compile(_RE_COLOR),
    'marker-end': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'marker-mid': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'marker-start': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'markerHeight': re.compile(_RE_NUMBER),
    'markerUnits': re.compile(r'^(strokeWidth|userSpaceOnUse)$'),
    'markerWidth': re.compile(_RE_NUMBER),
    'mask': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'opacity': re.compile(_RE_OPACITY),
    'orient': re.compile(r'^(auto|auto-start-reverse|-?\d+(\.\d+)?)$'),
    'overflow': re.compile(r'^(visible|hidden|scroll|auto)$'),
    'pointer-events': re.compile(r'^(visiblePainted|visibleFill|visibleStroke|visible|painted|fill|stroke|all|none)$'),
    'points': re.compile(_RE_NUMBER_LIST),
    'r': re.compile(_RE_NUMBER),
    'refX': re.compile(_RE_NUMBER),
    'refY': re.compile(_RE_NUMBER),
    'rx': re.compile(_RE_NUMBER),
    'ry': re.compile(_RE_NUMBER),
    'shape-rendering': re.compile(r'^(auto|optimizeSpeed|crispEdges|geometricPrecision)$'),
    'stop-color': re.compile(_RE_COLOR),
    'stop-opacity': re.compile(_RE_OPACITY),
    'stroke': re.compile(_RE_COLOR),
    'stroke-dasharray': re.compile(r'^(none|[\d\s,.]*)$'),
    'stroke-dashoffset': re.compile(_RE_NUMBER),
    'stroke-linecap': re.compile(r'^(butt|round|square)$'),
    'stroke-linejoin': re.compile(r'^(miter|round|bevel)$'),
    'stroke-miterlimit': re.compile(_RE_NUMBER),
    'stroke-opacity': re.compile(_RE_OPACITY),
    'stroke-width': re.compile(_RE_NUMBER),
    'text-anchor': re.compile(r'^(start|middle|end)$'),
    'text-decoration': re.compile(r'^(none|underline|overline|line-through|blink)$'),
    'text-rendering': re.compile(r'^(auto|optimizeSpeed|optimizeLegibility|geometricPrecision)$'),
    'transform': re.compile(_RE_TRANSFORM),
    'transform-origin': re.compile(_RE_TEXT),
    'unicode-bidi': re.compile(_RE_TEXT),
    'vector-effect': re.compile(r'^(none|non-scaling-stroke)$'),
    'visibility': re.compile(r'^(visible|hidden|collapse)$'),
    'word-spacing': re.compile(_RE_NUMBER),
    'writing-mode': re.compile(r'^(lr-tb|rl-tb|tb-rl|lr|rl|tb)$'),
    'width': re.compile(_RE_NUMBER),
    'height': re.compile(_RE_NUMBER),
    'viewBox': re.compile(_RE_VIEWBOX),
    'x': re.compile(_RE_NUMBER),
    'x1': re.compile(_RE_NUMBER),
    'x2': re.compile(_RE_NUMBER),
    'y': re.compile(_RE_NUMBER),
    'y1': re.compile(_RE_NUMBER),
    'y2': re.compile(_RE_NUMBER),
}

_CANONICAL_ATTRIBUTES = {s.lower(): s for s in _ALLOWED_ATTRIBUTES}

_DENIED_VALUE_PREFIXES = (
    'data:',
    'file:',
    'http:',
    'https:',
    'javascript:',
)


def _normalize(el: gws.XmlElement) -> Optional[gws.XmlElement]:
    name = _CANONICAL_TAGS.get(el.lcName)
    if name:
        return xmlx.tag(
            name,
            _normalize_atts(el.attrib),
            el.text if name in _TEXT_TAGS else None,
            gws.u.compact(_normalize(c) for c in el.children()))


def _normalize_atts(atts: dict) -> dict:
    res = {}
    for k, v in atts.items():
        # Skip if attribute is not in allowed list
        key = _CANONICAL_ATTRIBUTES.get(k.lower())
        if not key:
            continue

        val = str(v).strip()

        # Skip URLs that could lead to XSS
        if val.lower().startswith(_DENIED_VALUE_PREFIXES):
            continue

        # Validate attribute value against its regex pattern
        if _ALLOWED_ATTRIBUTES[key].match(val):
            res[key] = val

    return res
