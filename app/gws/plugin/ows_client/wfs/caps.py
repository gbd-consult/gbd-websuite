"""WFS capabilities parser."""

import gws
import gws.base.ows.client
import gws.lib.crs
import gws.base.ows.client.parseutil as u
import gws.gis.source
import gws.lib.xmlx as xmlx


# @TODO check support caps (we need at least BBOX)

def parse(xml) -> gws.OwsCapabilities:
    """Read WFS capabilities from the GetCapabilities XML.

    Each feature type is returned as a queryable source layer. Feature types
    without a supported CRS or a WGS84 extent get WGS84 defaults.

    Args:
        xml: GetCapabilities XML.

    Returns:
        The parsed capabilities.
    """
    caps_el = xmlx.from_string(xml, gws.XmlOptions(compactWhitespace=True, removeNamespaces=True))
    source_layers = gws.gis.source.check_layers(
        _feature_type(el) for el in caps_el.findall('FeatureTypeList/FeatureType'))
    return gws.OwsCapabilities(
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        sourceLayers=source_layers,
        version=caps_el.get('version'))


def _feature_type(type_el):
    """Create a source layer from a ``FeatureType`` element."""
    sl = gws.SourceLayer()

    sl.name = type_el.textof('Name')
    sl.title = type_el.textof('Title') or xmlx.namespace.plain_name(sl.name)
    sl.metadata = u.element_metadata(type_el)
    sl.isQueryable = True
    sl.supportedCrs = u.supported_crs(type_el) or [gws.lib.crs.WGS84]
    sl.wgsExtent = u.wgs_extent(type_el) or gws.lib.crs.WGS84.extent

    return sl
