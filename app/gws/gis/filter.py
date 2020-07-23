"""Search filters"""

import re

import gws

import gws.gis.bounds
import gws.gis.shape
import gws.tools.xml2
import gws.types as t


#:export
class SearchFilter(t.Data):
    operator: str
    name: str
    value: str
    shape: t.IShape
    sub: t.List['SearchFilter']


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


def from_fes_string(src: str) -> SearchFilter:
    try:
        el = gws.tools.xml2.from_string(src)
    except gws.tools.xml2.Error:
        raise Error('invalid xml')
    return from_fes_element(el)


def from_fes_element(el: gws.tools.xml2.Element) -> SearchFilter:
    op = el.name.lower()

    if op == 'filter':
        # root element, only allow a single child predicate
        if len(el.children) != 1:
            raise Error(f'invalid root predicate')
        return from_fes_element(el.first())

    if op in ('and', 'or'):
        return SearchFilter(operator=op, sub=[from_fes_element(c) for c in el.children])

    if op not in _SUPPORTED_OPS:
        raise Error(f'unsupported filter operation {op!r}')

    f = SearchFilter(
        operator=_SUPPORTED_OPS[op],
    )

    # @TODO support "prop = prop"

    v = el.first('ValueReference') or el.first('PropertyName')
    if not v or not v.text:
        raise Error(f'invalid property name')

    # we only support `propName` or `ns:propName`
    m = re.match(r'^(\w+:)?(\w+)$', v.text)
    if not m:
        raise Error(f'invalid property name {v.text!r}')
    f.name = m.group(2)

    if op == 'bbox':
        bounds = gws.gis.bounds.from_gml_envelope_element(el.first('Envelope'))
        if not bounds:
            raise Error(f'invalid BBOX')
        f.shape = gws.gis.shape.from_bounds(bounds)
        return f

    v = el.first('Literal')
    if v:
        f.value = v.text.strip()
        return f

    raise Error(f'unsupported filter')
