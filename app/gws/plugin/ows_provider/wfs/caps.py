import gws
import gws.lib.xml2 as xml2
import gws.gis.source
import gws.types as t

from .. import core
from .. import parseutil as u


# @TODO check support caps (we need at least BBOX)

def parse(xml) -> core.Caps:
    root_el = xml2.from_string(xml, compact_ws=True, strip_ns=True)
    source_layers = gws.gis.source.check_layers(
        _feature_type(e) for e in xml2.all(root_el, 'FeatureTypeList FeatureType'))
    return core.Caps(
        metadata=u.service_metadata(root_el),
        operations=u.service_operations(root_el),
        source_layers=source_layers,
        version=xml2.attr(root_el, 'version'))


def _feature_type(el):
    sl = gws.SourceLayer()

    sl.name = xml2.text(el, 'Name')
    sl.title = xml2.unqualify_name(sl.name)
    sl.metadata = u.element_metadata(el)
    sl.is_queryable = True
    sl.supported_bounds = u.supported_bounds(el)

    return sl
