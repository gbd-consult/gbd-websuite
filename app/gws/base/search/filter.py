"""OGC Filter Encoding 2.0 parser and matcher."""

import re
import operator

import gws
import gws.base.shape
import gws.lib.bounds
import gws.lib.gml
import gws.lib.xmlx as xmlx


class Error(gws.Error):
    """Invalid or unsupported filter."""

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
    """Evaluates a search filter against Python objects.

    By default, properties are object attributes and the geometry is the ``shape``
    attribute. Subclasses can override ``get_property`` and ``get_shape`` to match
    other kinds of objects.
    """

    def get_property(self, obj, prop):
        """Return a property value of an object.

        Args:
            obj: Object to match.
            prop: Property name.

        Returns:
            The attribute value, or ``None`` if the object has no such attribute.
        """
        return getattr(obj, prop, None)

    def get_shape(self, obj):
        """Return the geometry of an object.

        Args:
            obj: Object to match.

        Returns:
            The ``shape`` attribute, or ``None`` if the object has none.
        """
        return getattr(obj, 'shape', None)

    def matches(self, flt: gws.SearchFilter, obj):
        """Check if an object matches a filter.

        Calls the ``match_<operator>`` method for the filter operator.

        Args:
            flt: Search filter.
            obj: Object to match.

        Returns:
            ``True`` if the object matches the filter.

        Raises:
            ``AttributeError``: If the filter operator is not supported.
        """
        return getattr(self, f'match_{flt.operator}'.lower())(flt, obj)

    ##

    def match_and(self, flt, obj):
        """Check if an object matches all sub-filters.

        Args:
            flt: Search filter with the ``And`` operator.
            obj: Object to match.

        Returns:
            ``True`` if all sub-filters match.
        """
        return all(self.matches(sf, obj) for sf in flt.subFilters)

    def match_or(self, flt, obj):
        """Check if an object matches any sub-filter.

        Args:
            flt: Search filter with the ``Or`` operator.
            obj: Object to match.

        Returns:
            ``True`` if at least one sub-filter matches.
        """
        return any(self.matches(sf, obj) for sf in flt.subFilters)

    def match_not(self, flt, obj):
        """Check if an object does not match the first sub-filter.

        Args:
            flt: Search filter with the ``Not`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the first sub-filter does not match.
        """
        return not (self.matches(flt.subFilters[0], obj))

    ##

    def match_propertyisequalto(self, flt, obj):
        """Check if a property is equal to the filter value.

        Args:
            flt: Search filter with the ``PropertyIsEqualTo`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the property matches.
        """
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.eq)

    def match_propertyisnotequalto(self, flt, obj):
        """Check if a property is not equal to the filter value.

        Args:
            flt: Search filter with the ``PropertyIsNotEqualTo`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the property matches.
        """
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.ne)

    def match_propertyislessthan(self, flt, obj):
        """Check if a property is less than the filter value.

        Args:
            flt: Search filter with the ``PropertyIsLessThan`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the property matches.
        """
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.lt)

    def match_propertyisgreaterthan(self, flt, obj):
        """Check if a property is greater than the filter value.

        Args:
            flt: Search filter with the ``PropertyIsGreaterThan`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the property matches.
        """
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.gt)

    def match_propertyislessthanorequalto(self, flt, obj):
        """Check if a property is less than or equal to the filter value.

        Args:
            flt: Search filter with the ``PropertyIsLessThanOrEqualTo`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the property matches.
        """
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.le)

    def match_propertyisgreaterthanorequalto(self, flt, obj):
        """Check if a property is greater than or equal to the filter value.

        Args:
            flt: Search filter with the ``PropertyIsGreaterThanOrEqualTo`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the property matches.
        """
        return self.compare(self.get_property(obj, flt.property), flt.value, operator.ge)

    def compare(self, a, b, op):
        """Compare a property value with a filter value.

        If the property value is a list, it matches if any of its elements matches.

        Args:
            a: Property value.
            b: Filter value.
            op: Comparison function, like ``operator.eq``.

        Returns:
            ``True`` if the comparison succeeds, ``False`` if the property value is ``None``.
        """
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
        """Check if the object geometry intersects the filter box.

        Args:
            flt: Search filter with the ``BBOX`` operator.
            obj: Object to match.

        Returns:
            ``True`` if the geometry intersects the box, ``False`` if the object has no geometry.
        """
        shape = self.get_shape(obj)
        if not shape:
            return False
        return shape.intersects(flt.shape)


##


def from_fes_string(src: str) -> gws.SearchFilter:
    """Parse an FES filter from an XML string.

    Namespaces are removed before parsing.

    Args:
        src: XML string with a filter element.

    Returns:
        A search filter.

    Raises:
        ``Error``: If the XML is invalid or the filter is invalid or not supported.
    """
    try:
        el = xmlx.from_string(src, gws.XmlOptions(removeNamespaces=True))
    except Exception as exc:
        raise Error('invalid XML') from exc
    return from_fes_element(el)


def from_fes_element(el: gws.XmlElement) -> gws.SearchFilter:
    """Parse an FES filter from an XML element.

    The element can be a ``Filter`` root element with exactly one predicate, a logical
    operator (``And``, ``Or``, ``Not``) or a supported comparison or ``BBOX`` predicate.
    ``And`` and ``Or`` with a single operand are reduced to that operand. A comparison
    requires a ``ValueReference`` or ``PropertyName`` and a ``Literal``, ``BBOX``
    requires a property name and a GML ``Envelope``.

    Args:
        el: XML element.

    Returns:
        A search filter.

    Raises:
        ``Error``: If the filter is invalid or not supported.
    """
    op = el.name.lower()
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
