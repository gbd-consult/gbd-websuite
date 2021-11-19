# sanitizer
import gws
import gws.lib.xml2 as xml2
import gws.types as t
import gws.lib.image

_SVG_TAG_ATTS = {
    'xmlns': 'http://www.w3.org/2000/svg',
}


def fragment_to_element(fragment: t.List[gws.XmlElement], atts: dict = None) -> gws.XmlElement:
    fr = sorted(fragment, key=lambda el: el.attributes.get('z-index', 0))
    return xml2.tag('svg', _SVG_TAG_ATTS, atts, *fr)


def fragment_to_image(fragment: t.List[gws.XmlElement], size: gws.Size, format='png') -> gws.lib.image.Image:
    el = fragment_to_element(fragment)
    return gws.lib.image.from_svg(xml2.to_string(el), size, format)


def sanitize_element(el: gws.XmlElement) -> t.Optional[gws.XmlElement]:
    children = gws.compact(_sanitize(c) for c in el.children)
    if children:
        return xml2.tag('svg', _sanitize_atts(el.attributes), *children)


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

_ALLOWED_ATTRIBUTES = {
    'alignment-baseline',
    'baseline-shift',
    'clip',
    'clip-path',
    'clip-rule',
    'color',
    'color-interpolation',
    'color-interpolation-filters',
    'color-profile',
    'color-rendering',
    'cursor',
    'd',
    'direction',
    'display',
    'dominant-baseline',
    'enable-background',
    'fill',
    'fill-opacity',
    'fill-rule',
    'filter',
    'flood-color',
    'flood-opacity',
    'font-family',
    'font-size',
    'font-size-adjust',
    'font-stretch',
    'font-style',
    'font-variant',
    'font-weight',
    'glyph-orientation-horizontal',
    'glyph-orientation-vertical',
    'image-rendering',
    'kerning',
    'letter-spacing',
    'lighting-color',
    'marker-end',
    'marker-mid',
    'marker-start',
    'mask',
    'opacity',
    'overflow',
    'pointer-events',
    'shape-rendering',
    'stop-color',
    'stop-opacity',
    'stroke',
    'stroke-dasharray',
    'stroke-dashoffset',
    'stroke-linecap',
    'stroke-linejoin',
    'stroke-miterlimit',
    'stroke-opacity',
    'stroke-width',
    'text-anchor',
    'text-decoration',
    'text-rendering',
    'transform',
    'transform-origin',
    'unicode-bidi',
    'vector-effect',
    'visibility',
    'word-spacing',
    'writing-mode',
    'width',
    'height',
    'viewBox',
}


def _sanitize(el: gws.XmlElement) -> t.Optional[gws.XmlElement]:
    if el.name in _ALLOWED_TAGS:
        return xml2.element(
            el.name,
            _sanitize_atts(el.attributes),
            gws.compact(_sanitize(c) for c in el.children))

def _sanitize_atts(atts: dict) -> dict:
    res = {}
    for k, v in atts.items():
        if k not in _ALLOWED_ATTRIBUTES:
            continue
        if v.strip().startswith(('http:', 'https:', 'data:')):
            continue
        res[k] = v
    return res