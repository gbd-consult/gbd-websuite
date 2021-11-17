import gws
import gws.gis.ows.parseutil as u
import gws.lib.xml2
import gws.types as t

from .. import core


def parse(xml) -> core.Caps:
    root_el = gws.lib.xml2.from_string(xml)
    source_layers = u.enum_source_layers(_feature_type(e) for e in root_el.all('FeatureTypeList.FeatureType'))
    return core.Caps(
        metadata=u.get_service_metadata(root_el),
        operations=u.get_operations(root_el),
        source_layers=source_layers,
        version=root_el.attr('version'),
    )


def _feature_type(el):
    sl = gws.SourceLayer()

    sl.name = el.get_text('Name')
    if ':' in sl.name:
        _, _, sl.title = sl.name.rpartition(':')
    else:
        sl.title = sl.name

    sl.metadata = u.get_metadata(el)
    sl.is_queryable = True

    crs_tags = 'DefaultSRS', 'DefaultCRS', 'SRS', 'CRS', 'OtherSRS'
    extra_crsids = set(e.text for tag in crs_tags for e in el.all(tag))
    sl.supported_bounds = u.get_supported_bounds(el, extra_crsids)

    return sl
