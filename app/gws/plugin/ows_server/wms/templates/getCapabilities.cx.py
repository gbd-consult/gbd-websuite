import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl


def main(ta: server.TemplateArgs):
    if ta.version == '1.1.0':
        pass

    elif ta.version == '1.1.1':
        pass

    elif ta.version == '1.3.0':
        return tpl.to_xml(ta, ('WMS_Capabilities', doc13(ta)))


def doc13(ta: server.TemplateArgs):
    yield {
        'version': ta.version,
        'updateSequence': ta.service.updateSequence,
        'xmlns': 'wms',
    }
    yield 'Service', service_meta(ta), service_misc(ta)
    yield 'Capability', caps_13(ta)


def service_meta(ta: server.TemplateArgs):
    md = ta.service.metadata

    yield 'Name', md.name
    yield 'Title', md.title

    if md.abstract:
        yield 'Abstract', md.abstract

    yield tpl.keywords(md)
    yield tpl.online_resource(ta.serviceUrl)

    yield (
        'ContactInformation',
        (
            'ContactPersonPrimary',
            ('ContactPerson', md.contactPerson),
            ('ContactOrganization', md.contactOrganization)),
        ('ContactPosition', md.contactPosition),
        (
            'ContactAddress',
            ('AddressType', 'postal'),
            ('Address', md.contactAddress),
            ('City', md.contactCity),
            ('StateOrProvince', md.contactArea),
            ('PostCode', md.contactZip),
            ('Country', md.contactCountry)),
        ('ContactVoiceTelephone', md.contactPhone),
        ('ContactElectronicMailAddress', md.contactEmail))

    if md.fees:
        yield 'Fees', md.fees

    if md.accessConstraints:
        yield 'AccessConstraints', md.accessConstraints[0].title

    yield meta_links(ta, md)


def meta_links(ta: server.TemplateArgs, md: gws.Metadata):
    if md.metaLinks:
        for ml in md.metaLinks:
            yield (
                'MetadataURL', {'type': ml.type},
                ('Format', ml.format),
                tpl.online_resource(ta.url_for(ml.url)))


def service_misc(ta: server.TemplateArgs):
    # s = ta.service.layer_limit
    # if s:
    #     yield ('LayerLimit', s)
    #
    # s = ta.service.get('max_size')
    # if s:
    #     yield ('MaxWidth', s[0])
    #     yield ('MaxHeight', s[1])
    pass


def caps_13(ta: server.TemplateArgs):
    yield 'Request', request_caps(ta)

    yield 'Exception/Format', 'XML'

    if ta.service.withInspireMeta:
        yield 'inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)

    yield layer13(ta, ta.layerCapsTree.root)


def request_caps(ta: server.TemplateArgs):
    url = tpl.dcp_service_url(ta)

    for op in ta.service.supportedOperations:
        yield op.verb, [('Format', f) for f in op.formats], url


def layer13(ta: server.TemplateArgs, lc: server.LayerCaps):
    return 'Layer', {'queryable': 1 if lc.hasSearch else 0}, layer13_content(ta, lc)


def layer13_content(ta: server.TemplateArgs, lc: server.LayerCaps):
    md = lc.layer.metadata

    yield 'Name', lc.layerName
    yield 'Title', lc.title

    if md.abstract:
        yield 'Abstract', md.abstract

    yield tpl.keywords(md)

    if md.attribution and md.attribution.title:
        yield 'Attribution/Title', md.attribution.title

    if md.authorityUrl:
        yield 'AuthorityURL', {'name': md.authorityName}, tpl.online_resource(md.authorityUrl)

    if md.authorityIdentifier:
        yield 'Identifier', {'authority': md.authorityName}, md.authorityIdentifier

    yield meta_links(ta, md)

    for b in lc.bounds:
        yield 'CRS', b.crs.epsg

    yield (
        'EX_GeographicBoundingBox',
        ('westBoundLongitude', tpl.coord_dms(lc.layer.wgsExtent[0])),
        ('eastBoundLongitude', tpl.coord_dms(lc.layer.wgsExtent[2])),
        ('southBoundLatitude', tpl.coord_dms(lc.layer.wgsExtent[1])),
        ('northBoundLatitude', tpl.coord_dms(lc.layer.wgsExtent[3])),
    )

    for b in lc.bounds:
        if b.crs.isGeographic:
            minx, miny, maxx, maxy = b.extent[1], b.extent[0], b.extent[3], b.extent[2]
            f = tpl.coord_dms
        else:
            minx, miny, maxx, maxy = b.extent[0], b.extent[1], b.extent[2], b.extent[3]
            f = tpl.coord_m
        yield ('BoundingBox', {
            'CRS': b.crs.epsg,
            'minx': f(minx),
            'miny': f(miny),
            'maxx': f(maxx),
            'maxy': f(maxy),
        })

    if lc.hasLegend:
        yield 'Style', ('Name', 'default'), ('Title', 'default'), tpl.legend_url(ta, lc)

    if not lc.children:
        yield 'MinScaleDenominator', lc.minScale
        yield 'MaxScaleDenominator', lc.maxScale

    for c in reversed(lc.children):
        yield layer13(ta, c)
