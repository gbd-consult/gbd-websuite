import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def caps_11():
        pass

    def formats(verb):
        return [('Format', f) for f in ARGS.supported_formats[verb]]

    def meta_links(md):
        for ml in md.metaLinks:
            yield (
                'MetadataURL', {'type': ml.type},
                ('Format', ml.format),
                tpl.online_resource(ARGS.url_for(ml.url)))

    def service_meta():
        md = ARGS.service_meta

        yield 'Name', md.name
        yield 'Title', md.title

        if md.abstract:
            yield 'Abstract', md.abstract

        yield tpl.keywords(md)
        yield tpl.online_resource(ARGS.service_url)

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
            yield 'AccessConstraints', md.accessConstraints

        yield meta_links(md)

    def service_misc():
        # s = ARGS.service.layer_limit
        # if s:
        #     yield ('LayerLimit', s)
        #
        # s = ARGS.service.get('max_size')
        # if s:
        #     yield ('MaxWidth', s[0])
        #     yield ('MaxHeight', s[1])
        pass

    def layer13_content(lc):
        md = lc.meta

        yield 'Name', lc.layer_pname
        yield 'Title', lc.title

        if md.abstract:
            yield 'Abstract', md.abstract

        yield tpl.keywords(md)

        if md.attribution:
            yield 'Attribution Title', md.attribution

        if md.authorityUrl:
            yield 'AuthorityURL', {'name': md.authorityName}, tpl.online_resource(md.authorityUrl)

        if md.authorityIdentifier:
            yield 'Identifier', {'authority': md.authorityName}, md.authorityIdentifier

        yield meta_links(md)

        for b in lc.bounds:
            yield 'CRS', b.crs.epsg

        yield (
            'EX_GeographicBoundingBox',
            ('westBoundLongitude', lc.wgsExtent[0]),
            ('eastBoundLongitude', lc.wgsExtent[2]),
            ('southBoundLatitude', lc.wgsExtent[1]),
            ('northBoundLatitude', lc.wgsExtent[3]))

        for b in lc.bounds:
            yield ('BoundingBox', {
                'CRS': b.crs.epsg,
                'minx': b.extent[1] if b.crs.is_geographic else b.extent[0],
                'miny': b.extent[0] if b.crs.is_geographic else b.extent[1],
                'maxx': b.extent[3] if b.crs.is_geographic else b.extent[2],
                'maxy': b.extent[2] if b.crs.is_geographic else b.extent[3],
            })

        if lc.has_legend:
            yield 'Style', ('Name', 'default'), ('Title', 'default'), tpl.legendUrl(ARGS, lc)

        if not lc.children:
            yield 'MinScaleDenominator', lc.min_scale
            yield 'MaxScaleDenominator', lc.max_scale

        for c in reversed(lc.children):
            yield layer13(c)

    def layer13(lc):
        return 'Layer', {'queryable': 1 if lc.has_search else 0}, layer13_content(lc)

    def caps_13():
        url = tpl.dcp_service_url(ARGS)

        yield (
            'Request',
            ('GetCapabilities', formats('getcapabilities'), url),
            ('GetMap', formats('getmap'), url),
            ('GetFeatureInfo', formats('getfeatureinfo'), url),
            ('sld:GetLegendGraphic', formats('getlegendgraphic'), url))

        yield 'Exception Format', 'XML'

        if ARGS.with_inspire_meta:
            yield 'inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ARGS)

        yield layer13(ARGS.layer_root_caps)

    def doc13():
        yield {
            'version': ARGS.version,
            'updateSequence': ARGS.service.update_sequence,
            'xmlns': 'wms',
        }
        yield 'Service', service_meta(), service_misc()
        yield 'Capability', caps_13()

    ####

    if ARGS.version == '1.1.0':
        pass

    elif ARGS.version == '1.1.1':
        pass

    elif ARGS.version == '1.3.0':
        return tpl.to_xml(ARGS, ('WMS_Capabilities', doc13()))
