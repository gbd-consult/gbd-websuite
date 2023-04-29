import gws
import gws.base.ows
import gws.gis.crs
import gws.gis.ows.parseutil as u
import gws.gis.source
import gws.lib.xmlx as xmlx


# @TODO check support caps (we need at least BBOX)

def parse(xml) -> gws.base.ows.Caps:
    caps_el = xmlx.from_string(xml, compact_whitespace=True, remove_namespaces=True)
    source_layers = gws.gis.source.check_layers(
        _feature_type(el) for el in caps_el.findall('FeatureTypeList/FeatureType'))
    return gws.base.ows.Caps(
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        sourceLayers=source_layers,
        version=caps_el.get('version'))


def _feature_type(type_el):
    sl = gws.SourceLayer()

    sl.name = type_el.textof('Name')
    sl.title = type_el.textof('Title') or xmlx.namespace.unqualify(sl.name)
    sl.metadata = u.element_metadata(type_el)
    sl.isQueryable = True
    sl.supportedCrs = u.supported_crs(type_el) or [gws.gis.crs.WGS84]
    sl.wgsBounds = u.wgs_bounds(type_el) or gws.gis.crs.WGS84_BOUNDS

    return sl
