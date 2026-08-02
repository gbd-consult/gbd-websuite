# normalizer

from typing import Optional

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
    'stop',
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

# Regex patterns for attribute validation.
# Only the syntax of a value is validated, not its semantics.

_A = r'a-zA-Z'
_N = r'a-zA-Z0-9'
_P = r'+.,%'
_W = rf'{_N}_'

_ARGS = rf'[0-9{_P}\s-]+'
_KW = rf'[{_A}-]+'
_URL_REF = rf'url\(#[{_W}-]+\)'

_RE_COLOR = rf'^(#[{_N}]+|{_URL_REF}|{_KW}\({_ARGS}\)|{_KW})$'
_RE_FONT_FAMILY = rf'^[{_W}.,\s-]+$'
_RE_KEYWORD = rf'^{_KW}$'
_RE_NAME = rf'^[{_W}-]+$'
_RE_NAME_LIST = rf'^[{_W}\s-]*$'
_RE_NUMBER = rf'^[{_N}{_P}-]+$'
_RE_NUMBER_LIST = rf'^[{_N}{_P}\s-]+$'
_RE_PATH = rf'^[{_N},.\s-]+$'
_RE_TRANSFORM = rf'^{_KW}\({_ARGS}\)(\s{_KW}\({_ARGS}\))*$'
_RE_URL_REF = rf'^{_URL_REF}$'

# Dictionary of allowed attributes with their validation patterns

_ALLOWED_ATTRIBUTES: dict[str, str] = {
    'alignment-baseline': _RE_KEYWORD,
    'baseline-shift': _RE_NUMBER,
    'class': _RE_NAME_LIST,
    'clip': _RE_KEYWORD,
    'clip-path': _RE_URL_REF,
    'clip-rule': _RE_KEYWORD,
    'clipPathUnits': _RE_KEYWORD,
    'color': _RE_COLOR,
    'color-interpolation': _RE_KEYWORD,
    'color-interpolation-filters': _RE_KEYWORD,
    'color-profile': _RE_KEYWORD,
    'color-rendering': _RE_KEYWORD,
    'cursor': _RE_KEYWORD,
    'cx': _RE_NUMBER,
    'cy': _RE_NUMBER,
    'd': _RE_PATH,
    'direction': _RE_KEYWORD,
    'display': _RE_KEYWORD,
    'dominant-baseline': _RE_KEYWORD,
    'dx': _RE_NUMBER_LIST,
    'dy': _RE_NUMBER_LIST,
    'enable-background': _RE_KEYWORD,
    'fill': _RE_COLOR,
    'fill-opacity': _RE_NUMBER,
    'fill-rule': _RE_KEYWORD,
    'filter': _RE_URL_REF,
    'flood-color': _RE_COLOR,
    'flood-opacity': _RE_NUMBER,
    'font-family': _RE_FONT_FAMILY,
    'font-size': _RE_NUMBER,
    'font-size-adjust': _RE_NUMBER,
    'font-stretch': _RE_KEYWORD,
    'font-style': _RE_KEYWORD,
    'font-variant': _RE_KEYWORD,
    'font-weight': _RE_NUMBER,
    'fr': _RE_NUMBER,
    'fx': _RE_NUMBER,
    'fy': _RE_NUMBER,
    'glyph-orientation-horizontal': _RE_NUMBER,
    'glyph-orientation-vertical': _RE_NUMBER,
    'gradientTransform': _RE_TRANSFORM,
    'gradientUnits': _RE_KEYWORD,
    'hatchContentUnits': _RE_KEYWORD,
    'hatchUnits': _RE_KEYWORD,
    'id': _RE_NAME,
    'image-rendering': _RE_KEYWORD,
    'kerning': _RE_NUMBER,
    'lengthAdjust': _RE_KEYWORD,
    'letter-spacing': _RE_NUMBER,
    'lighting-color': _RE_COLOR,
    'marker-end': _RE_URL_REF,
    'marker-mid': _RE_URL_REF,
    'marker-start': _RE_URL_REF,
    'markerHeight': _RE_NUMBER,
    'markerUnits': _RE_KEYWORD,
    'markerWidth': _RE_NUMBER,
    'mask': _RE_URL_REF,
    'maskContentUnits': _RE_KEYWORD,
    'maskUnits': _RE_KEYWORD,
    'offset': _RE_NUMBER,
    'opacity': _RE_NUMBER,
    'orient': _RE_NUMBER,
    'overflow': _RE_KEYWORD,
    'pathLength': _RE_NUMBER,
    'patternContentUnits': _RE_KEYWORD,
    'patternTransform': _RE_TRANSFORM,
    'patternUnits': _RE_KEYWORD,
    'pitch': _RE_NUMBER,
    'pointer-events': _RE_KEYWORD,
    'points': _RE_NUMBER_LIST,
    'preserveAspectRatio': _RE_NAME_LIST,
    'r': _RE_NUMBER,
    'refX': _RE_NUMBER,
    'refY': _RE_NUMBER,
    'rotate': _RE_NUMBER_LIST,
    'rx': _RE_NUMBER,
    'ry': _RE_NUMBER,
    'shape-rendering': _RE_KEYWORD,
    'solid-color': _RE_COLOR,
    'solid-opacity': _RE_NUMBER,
    'spreadMethod': _RE_KEYWORD,
    'stop-color': _RE_COLOR,
    'stop-opacity': _RE_NUMBER,
    'stroke': _RE_COLOR,
    'stroke-dasharray': _RE_NUMBER_LIST,
    'stroke-dashoffset': _RE_NUMBER,
    'stroke-linecap': _RE_KEYWORD,
    'stroke-linejoin': _RE_KEYWORD,
    'stroke-miterlimit': _RE_NUMBER,
    'stroke-opacity': _RE_NUMBER,
    'stroke-width': _RE_NUMBER,
    'text-anchor': _RE_KEYWORD,
    'text-decoration': _RE_KEYWORD,
    'text-rendering': _RE_KEYWORD,
    'textLength': _RE_NUMBER,
    'transform': _RE_TRANSFORM,
    'transform-origin': _RE_NUMBER_LIST,
    'unicode-bidi': _RE_KEYWORD,
    'vector-effect': _RE_KEYWORD,
    'visibility': _RE_KEYWORD,
    'word-spacing': _RE_NUMBER,
    'writing-mode': _RE_KEYWORD,
    'width': _RE_NUMBER,
    'height': _RE_NUMBER,
    'viewBox': _RE_NUMBER_LIST,
    'x': _RE_NUMBER_LIST,
    'x1': _RE_NUMBER,
    'x2': _RE_NUMBER,
    'y': _RE_NUMBER_LIST,
    'y1': _RE_NUMBER,
    'y2': _RE_NUMBER,
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
        if re.match(_ALLOWED_ATTRIBUTES[key], val):
            res[key] = val

    return res
