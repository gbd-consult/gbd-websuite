"""OGC fes 2.0 filter

Supports

    - Minimum Standard Filter
        PropertyIsEqualTo, PropertyIsNotEqualTo, PropertyIsLessThan, PropertyIsGreaterThan,
        PropertyIsLessThanOrEqualTo, PropertyIsGreaterThanOrEqualTo.
        Implements the logical operators. Does not implement any additional functions.

    - Minimum Spatial Filter
        Implements only the BBOX spatial operator.

References:
    - OGC® Filter Encoding 2.0 Encoding Standard (http://docs.opengeospatial.org/is/09-026r2/09-026r2.html)
"""

import re
import operator

import gws
import gws.base.shape
import gws.lib.bounds
import gws.lib.gml
import gws.lib.xmlx as xmlx


class Error(gws.Error):
    pass


_SUPPORTED_OPS = {
    'propertyisequalto': gws.SearchFilterOperator.PropertyIsEqualTo,
    'propertyisnotequalto': gws.SearchFilterOperator.PropertyIsNotEqualTo,
    'propertyislessthan': gws.SearchFilterOperator.PropertyIsLessThan,
    'propertyisgreaterthan': gws.SearchFilterOperator.PropertyIsGreaterThan,
    'propertyislessthanorequalto': gws.SearchFilterOperator.PropertyIsLessThanOrEqualTo,
    'propertyisgreaterthanorequalto': gws.SearchFilterOperator.PropertyIsGreaterThanOrEqualTo,
    'bbox': gws.SearchFilterOperator.BBOX,
}


##

class Matcher:
    def get_property(self, obj, prop):
        return getattr(obj, prop, None)

    def get_shape(self, obj):
        return getattr(obj, 'shape', None)

    def matches(self, flt: gws.SearchFilter, obj):
        return getattr(self, f'match_{flt.operator}'.lower())(flt, obj)

    ##

    def match_and(self, flt, obj):
        return all(self.matches(sf, obj) for sf in flt.subFilters)

    def match_or(self, flt, obj):
        return any(self.matches(sf, obj) for sf in flt.subFilters)

    def match_not(self, flt, obj):
        return not (self.matches(flt.subFilters[0], obj))

    ##

    def match_propertyisequalto(self, flt, obj):
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.eq)

    def match_propertyisnotequalto(self, flt, obj):
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.ne)

    def match_propertyislessthan(self, flt, obj):
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.lt)

    def match_propertyisgreaterthan(self, flt, obj):
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.gt)

    def match_propertyislessthanorequalto(self, flt, obj):
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.le)

    def match_propertyisgreaterthanorequalto(self, flt, obj):
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.ge)

    def compare(self, a, b, op):
        if a is None:
            return False
        if isinstance(a, list):
            # @TODO matchAction
            return any(op(x, b) for x in a)
        return op(a, b)

    ##

    """
    @TODO
        Equals
        Disjoint
        Touches
        Within
        Overlaps
        Crosses
        Intersects
        Contains
        DWithin
        Beyond
    
    """

    def match_bbox(self, flt, obj):
        shape = self.get_shape(obj)
        if not shape:
            return False
        return shape.intersects(flt.shape)


##


def from_fes_string(src: str) -> gws.SearchFilter:
    try:
        el = xmlx.from_string(src, remove_namespaces=True)
    except Exception as exc:
        raise Error('invalid XML') from exc
    return from_fes_element(el)


def from_fes_element(el: gws.XmlElement) -> gws.SearchFilter:
    op = el.lcName
    sub = el.children()

    if op == 'filter':
        # root element, only allow a single child predicate
        if len(sub) != 1:
            raise Error(f'invalid root predicate')
        return from_fes_element(sub[0])

    if op == 'and':
        if len(sub) == 0:
            raise Error(f'invalid and predicate')
        if len(sub) == 1:
            return from_fes_element(sub[0])
        return gws.SearchFilter(operator=gws.SearchFilterOperator.And, subFilters=[from_fes_element(s) for s in sub])

    if op == 'or':
        if len(sub) == 0:
            raise Error(f'invalid or predicate')
        if len(sub) == 1:
            return from_fes_element(sub[0])
        return gws.SearchFilter(operator=gws.SearchFilterOperator.Or, subFilters=[from_fes_element(s) for s in sub])

    if op == 'not':
        if len(sub) != 1:
            raise Error(f'invalid not predicate')
        return gws.SearchFilter(operator=gws.SearchFilterOperator.Not, subFilters=[from_fes_element(s) for s in sub])

    if op not in _SUPPORTED_OPS:
        raise Error(f'unsupported filter operation {el.name!r}')

    flt = gws.SearchFilter(
        operator=_SUPPORTED_OPS[op],
    )

    # @TODO support "prop = prop"
    # @TODO support matchCase, matchAction

    v = el.findfirst('ValueReference', 'PropertyName')
    if not v or not v.text:
        raise Error(f'invalid property name or value reference')

    # we only support `propName` or `ns:propName`
    m = re.match(r'^(\w+:)?(\w+)$', v.text)
    if not m:
        raise Error(f'invalid property name {v.text!r}')
    flt.property = m.group(2)

    if op == 'bbox':
        v = el.findfirst('Envelope')
        if not v:
            raise Error(f'invalid envelope')
        bounds = gws.lib.gml.parse_envelope(v)
        flt.shape = gws.base.shape.from_bounds(bounds)
        return flt

    v = el.findfirst('Literal')
    if v:
        flt.value = v.text.strip()
        return flt

    raise Error(f'unsupported filter')
