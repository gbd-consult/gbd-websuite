"""Search filters"""

import re

import gws
import gws.gis.bounds
import gws.base.shape
import gws.lib.xmlx as xmlx
import gws.types as t


class Error(gws.Error):
    pass


"""
OGC fes 2.0 filter

http://docs.opengeospatial.org/is/09-026r2/09-026r2.html

Supports

    - Minimum Standard Filter
        PropertyIsEqualTo, PropertyIsNotEqualTo, PropertyIsLessThan, PropertyIsGreaterThan,
        PropertyIsLessThanOrEqualTo, PropertyIsGreaterThanOrEqualTo.
        Implements the logical operators. Does not implement any additional functions.

    - Minimum Spatial Filter
        Implements only the BBOX spatial operator.
"""

_SUPPORTED_OPS = {
    'propertyisequalto': '=',
    'propertyisnotequalto': '!=',
    'propertyislessthan': '<',
    'propertyisgreaterthan': '>',
    'propertyislessthanorequalto': '=<',
    'propertyisgreaterthanorequalto': '>=',
    'bbox': 'bbox',
}


def from_fes_string(src: str) -> gws.SearchFilter:
    try:
        el = xmlx.from_string(src)
    except Exception as exc:
        raise Error('invalid XML') from exc
    return from_fes_element(el)


def from_fes_element(el: gws.XmlElement) -> gws.SearchFilter:
    op = el.tag.lower()

    if op == 'filter':
        # root element, only allow a single child predicate
        if len(el) != 1:
            raise Error(f'invalid root predicate')
        return from_fes_element(el.children[0])

    if op in ('and', 'or'):
        return gws.SearchFilter(operator=op, sub=[from_fes_element(c) for c in el.children])

    if op not in _SUPPORTED_OPS:
        raise Error(f'unsupported filter operation {el.name!r}')

    f = gws.SearchFilter(
        operator=_SUPPORTED_OPS[op],
    )

    # @TODO support "prop = prop"

    v = el.first('ValueReference', 'PropertyName')
    if not v or not v.text:
        raise Error(f'invalid property name')

    # we only support `propName` or `ns:propName`
    m = re.match(r'^(\w+:)?(\w+)$', v.text)
    if not m:
        raise Error(f'invalid property name {v.text!r}')
    f.name = m.group(2)

    if op == 'bbox':
        v = el.first('Envelope')
        if v:
            bounds = gws.gis.bounds.from_gml_envelope_element()
            f.shape = gws.base.shape.from_bounds(bounds)
            return f

    v = el.first('Literal')
    if v:
        f.value = v.text.strip()
        return f

    raise Error(f'unsupported filter')
