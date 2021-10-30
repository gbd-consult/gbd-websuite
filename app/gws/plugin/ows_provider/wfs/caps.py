import gws
import gws.lib.gis
import gws.lib.metadata
import gws.lib.ows.parseutil as u
import gws.lib.gis
import gws.lib.xml2
import gws.types as t


class WFSCaps(gws.Data):
    metadata: gws.lib.metadata.Metadata
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str


def parse(xml) -> WFSCaps:
    root_el = gws.lib.xml2.from_string(xml)
    source_layers = u.flatten_source_layers(_feature_type(e) for e in root_el.all('FeatureTypeList.FeatureType'))

    return WFSCaps(
        metadata=u.get_service_metadata(root_el),
        operations=u.get_operations(root_el),
        source_layers=source_layers,
        supported_crs=gws.lib.gis.crs_from_source_layers(source_layers),
        version=root_el.attr('version'),
    )


def _feature_type(el):
    sl = gws.lib.gis.SourceLayer()

    n = el.get_text('Name')
    if ':' in n:
        sl.name = n
        sl.title = n.split(':')[1]
    else:
        sl.title = sl.name = n

    sl.metadata = u.get_metadata(el)
    sl.supported_bounds = u.get_bounds_list(el)

    sl.is_queryable = True

    cs = []
    e = u.one_of(el, 'DefaultSRS', 'DefaultCRS', 'SRS', 'CRS')
    if e:
        cs.append(e.text)

    cs.extend(e.text for e in el.all('OtherSRS'))
    cs.extend(e.text for e in el.all('OtherCRS'))

    sl.supported_crs = gws.compact(cs)

    return sl
