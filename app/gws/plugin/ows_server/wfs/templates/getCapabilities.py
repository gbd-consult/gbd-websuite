import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl


def main(args: dict):
    ta = tpl.TemplateArgs(args)
    return tpl.to_xml(ta, ('WFS_Capabilities', doc(ta)))


def doc(ta: tpl.TemplateArgs):
    yield {
        'version': ta.version,
        'xmlns': 'wfs',
    }

    # for lc in ta.layerCapsList:
    #     pfx, _ = tpl.split_name(lc.featureQname)
    #     yield {'xmlns:' + pfx: ''}

    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)

    yield 'ows:OperationsMetadata', operations(ta)

    yield (
        'FeatureTypeList',
        [('FeatureType', feature_type(ta, lc)) for lc in ta.layerCapsList]
    )

    yield 'fes:Filter_Capabilities', filters(ta)


def operations(ta: tpl.TemplateArgs, default_count=1000):
    versions = [tpl.ows_value(v) for v in ta.service.supportedVersions]

    yield (
        'ows:Operation',
        {'name': 'GetCapabilities'},
        tpl.ows_service_url(ta),
        ('ows:Parameter', {'name': 'AcceptVersions'}, ('ows:AllowedValues', versions)),
        ('ows:Parameter', {'name': 'AcceptFormats'}, ('ows:AllowedValues', tpl.ows_value('text/xml')))
    )

    yield (
        'ows:Operation',
        {'name': 'DescribeFeatureType'},
        tpl.ows_service_url(ta),
        ('ows:Parameter', {'name': 'OutputFormat'}, ('ows:AllowedValues', tpl.ows_value('text/xml; subtype=gml/3.2.1')))
    )

    yield (
        'ows:Operation',
        {'name': 'GetFeature'},
        tpl.ows_service_url(ta, get=True, post=True),
        ('ows:Parameter', {'name': 'OutputFormat'}, ('ows:AllowedValues', tpl.ows_value('text/xml; subtype=gml/3.2.1'))),
        ('ows:Parameter', {'name': 'ResultType'}, ('ows:AllowedValues', tpl.ows_value('results'), tpl.ows_value('hits')))
    )

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

    yield 'ows:Constraint', {'name': 'QueryExpressions'}, ('ows:AllowedValues', tpl.ows_value('wfs:Query'))

    if ta.service.withInspireMeta:
        yield 'ows:ExtendedCapabilities/inspire_dls:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)


def feature_type(ta: tpl.TemplateArgs, lc: server.LayerCaps):
    yield 'Name', lc.featureQname
    yield 'Title', lc.layer.title
    yield 'Abstract', lc.layer.metadata.abstract

    for n, b in enumerate(lc.bounds):
        if n == 0:
            yield 'DefaultCRS', b.crs.urn
        else:
            yield 'OtherCRS', b.crs.urn

    yield tpl.ows_wgs84_bounding_box(lc)

    if lc.layer.metadata.metaLinks:
        for ml in lc.layer.metadata.metaLinks:
            yield 'MetadataURL', {'xlink:href': ta.url_for(ml.url)}


def filters(ta: tpl.TemplateArgs):
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

    yield 'fes:Id_Capabilities/fes:ResourceIdentifier', {'name': 'fes:ResourceId'}

    yield (
        'fes:Spatial_Capabilities',
        ('fes:GeometryOperands/fes:GeometryOperand', {'name': 'gml:Envelope'}),
        ('fes:SpatialOperators/fes:SpatialOperator', {'name': 'BBOX'}),
    )


def constraint(name, value):
    ns, n = name.split(':')
    return ns + ':Constraint', {'name': n}, ('ows:NoValues',), ('ows:DefaultValue', value)
