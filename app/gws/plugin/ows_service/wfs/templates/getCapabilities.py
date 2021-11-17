import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def constraint(name, value):
        ns, n = name.split(':')
        return ns + ':Constraint', {'name': n}, ('ows:NoValues',), ('ows:DefaultValue', value)

    def val(value):
        return 'ows:Value', value

    def operations(default_count=1000):
        versions = [val(v) for v in ARGS.service.supported_versions]

        yield (
            'ows:Operation',
            {'name': 'GetCapabilities'},
            tpl.ows_service_url(ARGS),
            ('ows:Parameter', {'name': 'AcceptVersions'}, ('ows:AllowedValues', versions)),
            ('ows:Parameter', {'name': 'AcceptFormats'}, ('ows:AllowedValues', val('text/xml'))))

        yield (
            'ows:Operation',
            {'name': 'DescribeFeatureType'},
            tpl.ows_service_url(ARGS),
            ('ows:Parameter', {'name': 'OutputFormat'}, ('ows:AllowedValues', val('text/xml; subtype=gml/3.2.1'))))

        yield (
            'ows:Operation',
            {'name': 'GetFeature'},
            tpl.ows_service_url(ARGS, get=True, post=True),
            ('ows:Parameter', {'name': 'OutputFormat'}, ('ows:AllowedValues', val('text/xml; subtype=gml/3.2.1'))),
            ('ows:Parameter', {'name': 'ResultType'}, ('ows:AllowedValues', val('results'), val('hits'))))

        yield 'ows:Parameter', {'name': 'version'}, ('ows:AllowedValues', versions)

        yield (
            constraint('ows:ImplementsBasicWFS', 'TRUE'),
            constraint('ows:KVPEncoding', 'TRUE'),

            constraint('ows:ImplementsTransactionalWFS', 'FALSE'),
            constraint('ows:ImplementsLockingWFS', 'FALSE'),
            constraint('ows:XMLEncoding', 'FALSE'),
            constraint('ows:SOAPEncoding', 'FALSE'),
            constraint('ows:ImplementsInheritance', 'FALSE'),
            constraint('ows:ImplementsRemoteResolve', 'FALSE'),
            constraint('ows:ImplementsResultPaging', 'FALSE'),
            constraint('ows:ImplementsStandardJoins', 'FALSE'),
            constraint('ows:ImplementsSpatialJoins', 'FALSE'),
            constraint('ows:ImplementsTemporalJoins', 'FALSE'),
            constraint('ows:ImplementsFeatureVersioning', 'FALSE'),
            constraint('ows:ManageStoredQueries', 'FALSE'),

            constraint('ows:CountDefault', default_count),
        )

        yield 'ows:Constraint', {'name': 'QueryExpressions'}, ('ows:AllowedValues', val('wfs:Query'))

        if ARGS.with_inspire_meta:
            yield 'ows:ExtendedCapabilities inspire_dls:ExtendedCapabilities', tpl.inspire_extended_capabilities(ARGS)

    def filters():
        yield (
            'fes:Conformance',
            constraint('fes:ImplementsAdHocQuery', 'TRUE'),
            constraint('fes:ImplementsMinSpatialFilter', 'TRUE'),
            constraint('fes:ImplementsQuery', 'TRUE'),
            constraint('fes:ImplementsResourceId', 'TRUE'),
            constraint('fes:ImplementsMinStandardFilter', 'TRUE'),
            constraint('fes:ImplementsMinTemporalFilter', 'TRUE'),

            constraint('fes:ImplementsExtendedOperators', 'FALSE'),
            constraint('fes:ImplementsFunctions', 'FALSE'),
            constraint('fes:ImplementsMinimumXPath', 'FALSE'),
            constraint('fes:ImplementsSorting', 'FALSE'),
            constraint('fes:ImplementsSpatialFilter', 'FALSE'),
            constraint('fes:ImplementsStandardFilter', 'FALSE'),
            constraint('fes:ImplementsTemporalFilter', 'FALSE'),
            constraint('fes:ImplementsVersionNav', 'FALSE'),
        )

        yield 'fes:Id_Capabilities fes:ResourceIdentifier', {'name': 'fes:ResourceId'}

        yield (
            'fes:Spatial_Capabilities',
            ('fes:GeometryOperands fes:GeometryOperand', {'name': 'gml:Envelope'}),
            ('fes:SpatialOperators fes:SpatialOperator', {'name': 'BBOX'}),
        )

    def feature_type(lc):
        yield 'Name', lc.feature_qname
        yield 'Title', lc.title
        yield 'Abstract', lc.meta.abstract

        for n, b in enumerate(lc.bounds):
            if n == 0:
                yield 'DefaultCRS', b.crs.urn
            else:
                yield 'OtherCRS', b.crs.urn

        yield tpl.ows_wgs84_bounding_box(lc)

        for ml in lc.meta.metaLinks:
            yield 'MetadataURL', {'xlink:href': ARGS.url_for(ml.url)}

    def caps():
        yield {
            'version': ARGS.version,
            'xmlns': 'wfs',
        }

        for lc in ARGS.layer_caps_list:
            pfx, _ = tpl.split_name(lc.feature_qname)
            yield {'xmlns:' + pfx: ''}

        yield tpl.ows_service_identification(ARGS)
        yield tpl.ows_service_provider(ARGS)

        yield 'ows:OperationsMetadata', operations()

        yield (
            'FeatureTypeList',
            [('FeatureType', feature_type(lc)) for lc in ARGS.layer_caps_list]
        )

        yield 'fes:Filter_Capabilities', filters()

    ##

    return tpl.to_xml(ARGS, ('WFS_Capabilities', caps()))
