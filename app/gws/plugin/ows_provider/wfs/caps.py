import gws
import gws.lib.xml3 as xml3
import gws.gis.source
import gws.types as t

from .. import core
from .. import parseutil as u


# @TODO check support caps (we need at least BBOX)

def parse(xml) -> core.Caps:
    root_el = xml3.from_string(xml, compact_ws=True, strip_ns=True)
    source_layers = gws.gis.source.check_layers(
        _feature_type(e) for e in xml3.all(root_el, 'FeatureTypeList.FeatureType'))
    return core.Caps(
        metadata=u.service_metadata(root_el),
        operations=u.service_operations(root_el),
        source_layers=source_layers,
        version=xml3.attr(root_el, 'version'))


def _feature_type(el):
    sl = gws.SourceLayer()

    sl.name = xml3.text(el, 'Name')
    sl.title = xml3.unqualify_name(sl.name)
    sl.metadata = u.element_metadata(el)
    sl.is_queryable = True
    sl.supported_bounds = u.supported_bounds(el)

    return sl
