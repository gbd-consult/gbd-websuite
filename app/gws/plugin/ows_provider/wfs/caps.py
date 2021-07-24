import gws.lib.gis
import gws.lib.metadata
import gws.lib.ows.parseutil as u
import gws.lib.gis
import gws.lib.xml2
import gws.types as t


class WFSCaps(gws.Data):
    metadata: gws.lib.metadata.Record
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str


def parse(xml) -> WFSCaps:
    el = gws.lib.xml2.from_string(xml)

    meta = u.get_meta(u.one_of(el, 'Service', 'ServiceIdentification'))
    meta['contact'] = u.get_meta_contact(u.one_of(el, 'Service.ContactInformation', 'ServiceProvider.ServiceContact'))
    meta['url'] = u.get_url(el.first('ServiceMetadataURL'))

    source_layers = u.flatten_source_layers(_feature_type(e) for e in el.all('FeatureTypeList.FeatureType'))

    return WFSCaps(
        metadata=gws.lib.metadata.Record(meta),
        operations=[gws.OwsOperation(e) for e in u.get_operations(u.one_of(el, 'OperationsMetadata', 'Capability'))],
        source_layers=source_layers,
        supported_crs=gws.lib.gis.crs_from_layers(source_layers),
        version=el.attr('version'),
    )


def _feature_type(el):
    oo = gws.lib.gis.SourceLayer()

    n = el.get_text('Name')
    if ':' in n:
        oo.name = n
        oo.title = n.split(':')[1]
    else:
        oo.title = oo.name = n

    oo.metadata = gws.lib.metadata.Record(u.get_meta(el))
    oo.supported_bounds = u.get_bounds_list(el)

    oo.is_queryable = True

    cs = []
    e = u.one_of(el, 'DefaultSRS', 'DefaultCRS', 'SRS', 'CRS')
    if e:
        cs.append(e.text)

    cs.extend(e.text for e in el.all('OtherSRS'))
    cs.extend(e.text for e in el.all('OtherCRS'))

    oo.supported_crs = gws.compact(cs)

    return oo
