# sanitizer

from typing import Optional

import re
from typing import Optional, Dict, Pattern

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


def sanitize_element(el: gws.XmlElement) -> Optional[gws.XmlElement]:
    """Remove unsafe stuff from an SVG element."""

    children = gws.u.compact(_sanitize(c) for c in el)
    if children:
        return xmlx.tag('svg', _sanitize_atts(el.attrib), *children)


##

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

# Regex patterns for attribute validation
_RE_COLOR = r'^(#[0-9A-Fa-f]{3,8}|(rgb|rgba|hsl|hsla)\([\d%,.\s]+\)|aliceblue|antiquewhite|aqua|aquamarine|azure|beige|bisque|black|blanchedalmond|blue|blueviolet|brown|burlywood|cadetblue|chartreuse|chocolate|coral|cornflowerblue|cornsilk|crimson|cyan|darkblue|darkcyan|darkgoldenrod|darkgray|darkgreen|darkgrey|darkkhaki|darkmagenta|darkolivegreen|darkorange|darkorchid|darkred|darksalmon|darkseagreen|darkslateblue|darkslategray|darkslategrey|darkturquoise|darkviolet|deeppink|deepskyblue|dimgray|dimgrey|dodgerblue|firebrick|floralwhite|forestgreen|fuchsia|gainsboro|ghostwhite|gold|goldenrod|gray|green|greenyellow|grey|honeydew|hotpink|indianred|indigo|ivory|khaki|lavender|lavenderblush|lawngreen|lemonchiffon|lightblue|lightcoral|lightcyan|lightgoldenrodyellow|lightgray|lightgreen|lightgrey|lightpink|lightsalmon|lightseagreen|lightskyblue|lightslategray|lightslategrey|lightsteelblue|lightyellow|lime|limegreen|linen|magenta|maroon|mediumaquamarine|mediumblue|mediumorchid|mediumpurple|mediumseagreen|mediumslateblue|mediumspringgreen|mediumturquoise|mediumvioletred|midnightblue|mintcream|mistyrose|moccasin|navajowhite|navy|oldlace|olive|olivedrab|orange|orangered|orchid|palegoldenrod|palegreen|paleturquoise|palevioletred|papayawhip|peachpuff|peru|pink|plum|powderblue|purple|red|rosybrown|royalblue|saddlebrown|salmon|sandybrown|seagreen|seashell|sienna|silver|skyblue|slateblue|slategray|slategrey|snow|springgreen|steelblue|tan|teal|thistle|tomato|turquoise|violet|wheat|white|whitesmoke|yellow|yellowgreen|transparent|currentColor)$'
_RE_NUMBER = r'^-?\d+(\.\d+)?(px|em|ex|pt|pc|cm|mm|in|%)?$'
_RE_OPACITY = r'^(0(\.\d+)?|1(\.0+)?)$'
_RE_PATH = r'^[mMlLhHvVcCsSqQtTaAzZ0-9\s,.-]+$'
_RE_TRANSFORM = r'^(matrix|translate|scale|rotate|skewX|skewY)\([\d\s,.-]+\)( (matrix|translate|scale|rotate|skewX|skewY)\([\d\s,.-]+\))*$'
_RE_VIEWBOX = r'^\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?$'
_RE_TEXT = r'^[^<>]*$'
_RE_FONT_FAMILY = r'^[^<>"\']*$'
_RE_ANY = r'.*'

# Dictionary of allowed attributes with their validation patterns
_ALLOWED_ATTRIBUTES: Dict[str, Pattern] = {
    'alignment-baseline': re.compile(_RE_TEXT),
    'baseline-shift': re.compile(_RE_TEXT),
    'clip': re.compile(_RE_TEXT),
    'clip-path': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'clip-rule': re.compile(r'^(nonzero|evenodd)$'),
    'color': re.compile(_RE_COLOR),
    'color-interpolation': re.compile(r'^(auto|sRGB|linearRGB)$'),
    'color-interpolation-filters': re.compile(r'^(auto|sRGB|linearRGB)$'),
    'color-profile': re.compile(_RE_TEXT),
    'color-rendering': re.compile(r'^(auto|optimizeSpeed|optimizeQuality)$'),
    'cursor': re.compile(_RE_TEXT),
    'd': re.compile(_RE_PATH),
    'direction': re.compile(r'^(ltr|rtl)$'),
    'display': re.compile(r'^(inline|block|list-item|run-in|compact|marker|table|inline-table|table-row-group|table-header-group|table-footer-group|table-row|table-column-group|table-column|table-cell|table-caption|none)$'),
    'dominant-baseline': re.compile(_RE_TEXT),
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
    'image-rendering': re.compile(r'^(auto|optimizeSpeed|optimizeQuality)$'),
    'kerning': re.compile(_RE_TEXT),
    'letter-spacing': re.compile(_RE_NUMBER),
    'lighting-color': re.compile(_RE_COLOR),
    'marker-end': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'marker-mid': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'marker-start': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'mask': re.compile(r'^url\(#[a-zA-Z0-9_-]+\)$'),
    'opacity': re.compile(_RE_OPACITY),
    'overflow': re.compile(r'^(visible|hidden|scroll|auto)$'),
    'pointer-events': re.compile(r'^(visiblePainted|visibleFill|visibleStroke|visible|painted|fill|stroke|all|none)$'),
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
}


def _sanitize(el: gws.XmlElement) -> Optional[gws.XmlElement]:
    if el.name in _ALLOWED_TAGS:
        return xmlx.tag(
            el.name,
            _sanitize_atts(el.attrib),
            gws.u.compact(_sanitize(c) for c in el.children()))


def _sanitize_atts(atts: dict) -> dict:
    res = {}
    for k, v in atts.items():
        # Skip if attribute is not in allowed list
        if k not in _ALLOWED_ATTRIBUTES:
            continue
        
        # Skip URLs that could lead to XSS
        if v.strip().startswith(('http:', 'https:', 'data:')):
            continue
        
        # Validate attribute value against its regex pattern
        if _ALLOWED_ATTRIBUTES[k].match(v.strip()):
            res[k] = v
    
    return res
