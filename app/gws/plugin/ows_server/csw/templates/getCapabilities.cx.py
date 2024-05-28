import gws.base.ows.server.templatelib as tpl


def main(ARGS):
    def doc_iso():
        yield {'xmlns': 'csw', 'version': ARGS.version}

        yield tpl.ows_service_identification(ARGS)
        yield tpl.ows_service_provider(ARGS)

        yield (
            'ows:OperationsMetadata',
            ('ows:Operation', {'name': 'GetCapabilities'}, tpl.ows_service_url(ARGS)),
            (
                'ows:Operation',
                {'name': 'GetRecords'},
                tpl.ows_service_url(ARGS, post=True),
                ('ows:Parameter', {'name': 'typeName'}, ('ows:AllowedValues ows:Value', 'csw:Record')),
                ('ows:Parameter', {'name': 'outputFormat'}, ('ows:AllowedValues ows:Value', 'application/xml')),
                ('ows:Parameter', {'name': 'outputSchema'}, ('ows:AllowedValues ows:Value', 'http://www.opengis.net/cat/csw/2.0.2')),
                ('ows:Parameter', {'name': 'resultType'}, ('ows:AllowedValues ows:Value', 'results')),
                ('ows:Parameter', {'name': 'ElementSetName'}, ('ows:AllowedValues ows:Value', 'full')),
                ('ows:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, ('ows:AllowedValues ows:Value', 'Filter')),
                ('ows:Parameter', {'name': 'version'}, ('ows:AllowedValues ows:Value', ARGS.version))
            ),
            (
                'ows:Operation',
                {'name': 'DescribeRecord'},
                tpl.ows_service_url(ARGS),
                ('ows:Parameter', {'name': 'typeName'}, ('ows:AllowedValues ows:Value', 'gmd:MD_Metadata')),
                ('ows:Parameter', {'name': 'outputFormat'}, ('ows:AllowedValues ows:Value', 'application/xml')),
                ('ows:Parameter', {'name': 'schemaLanguage'}, ('ows:AllowedValues ows:Value', 'http://www.w3.org/XML/Schema')),
                ('ows:Parameter', {'name': 'resultType'}, ('ows:AllowedValues ows:Value', 'results')),
                ('ows:Parameter', {'name': 'ElementSetName'}, ('ows:AllowedValues ows:Value', 'full')),
                ('ows:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, ('ows:AllowedValues ows:Value', 'Filter')),
                ('ows:Parameter', {'name': 'version'}, ('ows:AllowedValues ows:Value', ARGS.version))
            ),

            ('ows:Constraint', {'name': 'IsoProfiles'}, ('ows:AllowedValues ows:Value', 'http://www.isotc211.org/2005/gmd')),
            ('inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ARGS))
        )

        yield (
            'ogc:Filter_Capabilities',
            (
                'ogc:Spatial_Capabilities',
                ('ogc:Spatial_Operators ogc:BBOX', '')),
            (
                'ogc:Scalar_Capabilities',
                ('ogc:Logical_Operators', ''),
                (
                    'ogc:Comparison_Operators',
                    ('ogc:Simple_Comparisons', ''),
                    ('ogc:Like', ''),
                    ('ogc:NullCheck', ''))
            )
        )

    if ARGS.profile == 'ISO':
        return tpl.to_xml(ARGS, ('Capabilities', doc_iso()))
