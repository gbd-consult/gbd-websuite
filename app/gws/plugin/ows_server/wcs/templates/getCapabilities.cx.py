import gws.base.ows.server.templatelib as tpl


def main(ARGS):
    def service1():
        md = ARGS.service_meta

        yield 'Name', ARGS.service.name
        yield 'Title', md.title

        if md.abstract:
            yield 'Abstract', md.abstract

        yield tpl.wms_keywords(md)

        if md.fees:
            yield 'Fees', md.fees
        if md.accessConstraints:
            yield 'AccessConstraints', md.accessConstraints

        yield (
            'responsibleParty',
            ('individualName', md.contactPerson),
            ('organisationName', md.contactOrganization),
            ('positionName', md.contactPosition),
            (
                'contactInfo',
                ('phone voice', md.contactPhone),
                ('deliveryPoint', md.contactAddress),
                ('city', md.contactCity),
                ('administrativeArea', md.contactArea),
                ('postalCode', md.contactZip),
                ('country', md.contactCountry),
                ('electronicMailAddress', md.contactEmail),
            )
        )

    def content1():
        for lc in ARGS.layer_caps_list:
            yield (
                'CoverageOfferingBrief',
                ('label', lc.title),
                ('name', lc.layer_pname),
                tpl.lon_lat_envelope(lc))

    def doc1():
        yield {
            'xmlns': 'wcs',
            'version': ARGS.version,
        }

        yield 'Service', service1()

        yield (
            'Capability Request',
            ('GetCapabilities', tpl.dcp_service_url(ARGS)),
            ('DescribeCoverage', tpl.dcp_service_url(ARGS)),
            ('GetCoverage', tpl.dcp_service_url(ARGS)))

        yield 'ContentMetadata', content1()

    def content2():
        for lc in ARGS.layer_caps_list:
            yield (
                'CoverageSummary',
                ('CoverageId', lc.layer_pname),
                ('CoverageSubtype', 'RectifiedGridCoverage'),
                ('Title', lc.title),
                ('Abstract', lc.meta.abstract),
                tpl.ows_wgs84_bounding_box(lc))

    def doc2():
        yield {
            'xmlns': 'wcs',
            'version': ARGS.version,
        }

        yield tpl.ows_service_identification(ARGS)
        yield tpl.ows_service_provider(ARGS)

        yield (
            'ows:OperationsMetadata',

            ('ows:Operation', {'name': 'GetCapabilities'}, tpl.ows_service_url(ARGS)),
            ('ows:Operation', {'name': 'DescribeCoverage'}, tpl.ows_service_url(ARGS)),
            ('ows:Operation', {'name': 'GetCoverage'}, tpl.ows_service_url(ARGS)))

        yield 'ServiceMetadata formatSupported', 'image/png'
        yield 'Contents', content2()

    ##

    if ARGS.version.startswith('1'):
        return tpl.to_xml_response(ARGS, ('WCS_Capabilities', doc1()))

    if ARGS.version.startswith('2'):
        return tpl.to_xml_response(ARGS, ('Capabilities', doc2()))
