import gws
import gws.lib.xmlx as xmlx
import gws.gis.source
import gws.types as t

from .. import core
from .. import parseutil as u


# @TODO check support caps (we need at least BBOX)

def parse(xml) -> core.Caps:
    caps_el = xmlx.from_string(xml, compact_whitespace=True, remove_namespaces=True)
    source_layers = gws.gis.source.check_layers(
        _feature_type(el) for el in caps_el.findall('FeatureTypeList/FeatureType'))
    return core.Caps(
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        source_layers=source_layers,
        version=caps_el.get('version'))


def _feature_type(type_el):
    sl = gws.SourceLayer()

    sl.name = type_el.text_of('Name')
    sl.title = xmlx.namespace.unqualify(sl.name)
    sl.metadata = u.element_metadata(type_el)
    sl.isQueryable = True
    sl.supported_bounds = u.supported_bounds(type_el)

    return sl
