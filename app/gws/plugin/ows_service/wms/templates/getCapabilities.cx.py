from gws.lib.xml3 import tag, with_xmlns


def online_resource(url):
    return tag('OnlineResource', {
        'xlink:type': 'simple',
        'xlink:href': url
    })


def dcp_service_url(ARGS):
    # OGC 01-068r3, 6.2.2
    # The URL prefix shall end in either a '?' (in the absence of additional server-specific parameters) or a '&'.
    # OGC 06-042, 6.3.3
    # A URL prefix is defined... as a string including... mandatory question mark
    return tag('DCPType HTTP Get', online_resource(ARGS.service_url + '?'))


def legend_url(ARGS, layer_caps):
    return tag('LegendURL',
               tag('Format', 'image/png'),
               online_resource(ARGS.service_url + '?request=GetLegendGraphic&layer=' + layer_caps.layer_xname.p))


def inspire_extended_capabilities(ARGS):
    return 'hi'


def keywords(md, ows=False):
    if ows:
        container = 'ows:Keywords'
        tg = 'ows:Keyword'
    else:
        container = 'KeywordList'
        tg = 'Keyword'

    kws = []

    if md.keywords:
        for kw in md.keywords:
            kws.append(tag(tg, kw))
    if md.inspireTheme:
        kws.append(tag(tg, md.inspireThemeNameEn, vocabulary="GEMET - INSPIRE themes"))
    if md.isoTopicCategory:
        kws.append(tag(tg, md.isoTopicCategory, vocabulary="ISO 19115:2003"))
    if md.inspireMandatoryKeyword:
        kws.append(tag(tg, md.inspireMandatoryKeyword, vocabulary="ISO"))

    if kws:
        return tag(container, kws)


##


def main(ARGS):
    def caps_11():
        pass

    def formats(verb):
        for f in ARGS.supported_formats[verb]:
            yield tag('Format', f)

    def service_meta():
        md = ARGS.service_meta

        yield tag('Name', md.name)
        yield tag('Title', md.title)

        if md.abstract:
            yield tag('Abstract', md.abstract)

        yield keywords(md)
        yield online_resource(ARGS.service_url)

        yield tag('ContactInformation',
                  tag('ContactPersonPrimary',
                      tag('ContactPerson', md.contactPerson),
                      tag('ContactOrganization', md.contactOrganization)),
                  tag('ContactPosition', md.contactPosition),
                  tag('ContactAddress',
                      tag('AddressType', 'postal'),
                      tag('Address', md.contactAddress),
                      tag('City', md.contactCity),
                      tag('StateOrProvince', md.contactArea),
                      tag('PostCode', md.contactZip),
                      tag('Country', md.contactCountry)),
                  tag('ContactVoiceTelephone', md.contactPhone),
                  tag('ContactElectronicMailAddress', md.contactEmail))

        if md.fees:
            yield tag('Fees', md.fees)
        if md.accessConstraints:
            yield tag('AccessConstraints', md.accessConstraints)

    def service_misc():
        # s = ARGS.service.layer_limit
        # if s:
        #     yield tag('LayerLimit', s)
        #
        # s = ARGS.service.get('max_size')
        # if s:
        #     yield tag('MaxWidth', s[0])
        #     yield tag('MaxHeight', s[1])
        pass

    def inspire():
        if ARGS.with_inspire_meta:
            yield tag('inspire_vs:ExtendedCapabilities', inspire_extended_capabilities(ARGS))

    def layer_13_meta(lc):
        yield tag('Name', lc.layer_xname.p)
        yield tag('Title', lc.title)

        if lc.meta.abstract:
            yield tag('Abstract', lc.meta.abstract)

        yield keywords(lc.meta)

        if lc.meta.attribution:
            yield tag('Attribution Title', lc.meta.attribution)

        if lc.meta.authorityUrl:
            yield tag('AuthorityURL', online_resource(lc.meta.authorityUrl), name=lc.meta.authorityName)

        if lc.meta.authorityIdentifier:
            yield tag('Identifier', lc.meta.authorityIdentifier, authority=lc.meta.authorityName)

        # @wms_meta_links lc.meta

    def layer_13_content(lc):
        yield layer_13_meta(lc)

        for b in lc.bounds:
            yield tag('CRS', b.crs.epsg)

        yield tag('EX_GeographicBoundingBox',
                  tag('westBoundLongitude', lc.extent4326[0]),
                  tag('eastBoundLongitude', lc.extent4326[2]),
                  tag('southBoundLatitude', lc.extent4326[1]),
                  tag('northBoundLatitude', lc.extent4326[3]))

        for b in lc.bounds:
            yield tag('BoundingBox', {
                'CRS': b.crs.epsg,
                'minx': b.extent[1] if b.crs.is_geographic else b.extent[0],
                'miny': b.extent[0] if b.crs.is_geographic else b.extent[1],
                'maxx': b.extent[3] if b.crs.is_geographic else b.extent[2],
                'maxy': b.extent[2] if b.crs.is_geographic else b.extent[3],
            })

        if lc.has_legend:
            yield tag('Style',
                      tag('Name', 'default'),
                      tag('Title', 'default'),
                      legend_url(ARGS, lc))

        if not lc.children:
            yield tag('MinScaleDenominator', lc.min_scale)
            yield tag('MaxScaleDenominator', lc.max_scale)

        for c in reversed(lc.children):
            yield layer_13(c)

    def layer_13(lc):
        return tag('Layer', {'queryable': 1 if lc.has_search else 0}, layer_13_content(lc))

    def caps_13():
        url = dcp_service_url(ARGS)
        atts = {
            'version': ARGS.version,
            'updateSequence': ARGS.service.update_sequence,
            'xmlns': 'wms',
        }

        yield tag('WMS_Capabilities', atts,
                  tag('Service',
                      service_meta(),
                      service_misc()),
                  tag('Capability',
                      tag('Request',
                          tag('GetCapabilities', formats('getcapabilities'), url),
                          tag('GetMap', formats('getmap'), url),
                          tag('GetFeatureInfo', formats('getfeatureinfo'), url),
                          tag('sld:GetLegendGraphic', formats('getlegendgraphic'), url)),
                      tag('Exception Format', 'XML'),
                      inspire(),
                      layer_13(ARGS.layer_root_caps)

                      )
                  )

    ####

    if ARGS.version == '1.1.0':
        return caps_11()

    elif ARGS.version == '1.1.1':
        return caps_11()

    elif ARGS.version == '1.3.0':
        return next(caps_13())
