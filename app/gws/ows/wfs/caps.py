import gws
import gws.ows.parseutil as u
import gws.tools.xml3
import gws.types as t

from . import types


def parse(srv: t.ServiceInterface, xml):
    el = gws.tools.xml3.from_string(xml)

    srv.meta = t.MetaData(u.get_meta(
        u.one_of(el, 'Service', 'ServiceIdentification')))

    srv.meta.contact = t.MetaContact(u.get_meta_contact(
        u.one_of(el, 'Service.ContactInformation', 'ServiceProvider.ServiceContact')))

    if not srv.meta.url:
        srv.meta.url = u.get_url(el.first('ServiceMetadataURL'))

    srv.operations = u.get_operations(
        u.one_of(el, 'OperationsMetadata', 'Capability'))

    srv.version = el.attr('version')
    srv.layers = u.flatten_source_layers(_feature_type(e) for e in el.all('FeatureTypeList.FeatureType'))
    srv.supported_crs = u.crs_from_layers(srv.layers)


def _feature_type(el):
    oo = types.SourceLayer()

    n = el.get_text('Name')
    if ':' in n:
        oo.name = n
        oo.title = n.split(':')[1]
    else:
        oo.title = oo.name = n

    oo.meta = t.MetaData(u.get_meta(el))
    oo.extents = u.get_extents(el)

    oo.is_queryable = True

    cs = []
    e = u.one_of(el, 'DefaultSRS', 'DefaultCRS', 'SRS', 'CRS')
    if e:
        cs.append(e.text)

    cs.extend(e.text for e in el.all('OtherSRS'))
    cs.extend(e.text for e in el.all('OtherCRS'))

    oo.supported_crs = gws.compact(cs)

    return oo
