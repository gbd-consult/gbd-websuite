"""WFS GetCapabilities template."""

import gws
import gws.lib.xmlx
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        (
            'WFS_Capabilities',
            {'version': ta.version},
            doc(ta),
        ),
        namespaces={
            'ows': gws.lib.xmlx.namespace.require('ows11'),
            'gml': gws.lib.xmlx.namespace.require('gml2'),
            **tpl.namespaces_from_caps(ta),
        },
        default_namespace=gws.lib.xmlx.namespace.require('wfs'),
    )


def doc(ta: server.TemplateArgs):
    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)
    yield 'ows:OperationsMetadata', operations(ta)
    yield 'FeatureTypeList', feature_type_list(ta)
    yield 'fes:Filter_Capabilities', filters(ta)


def operations(ta: server.TemplateArgs):
    versions = [tpl.ows_value(v) for v in ta.service.supportedVersions]

    for op in ta.service.supportedOperations:
        yield (
            'ows:Operation',
            {'name': op.verb},
            tpl.ows_service_url(ta),
            operation_params(ta, op),
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
        constraint('ows:ImplementsResultPaging', 'TRUE'),
        constraint('ows:ImplementsStandardJoins', 'FALSE'),
        constraint('ows:ImplementsSpatialJoins', 'FALSE'),
        constraint('ows:ImplementsTemporalJoins', 'FALSE'),
        constraint('ows:ImplementsFeatureVersioning', 'FALSE'),
        constraint('ows:ManageStoredQueries', 'FALSE'),
        constraint('ows:CountDefault', ta.service.maxFeatureCount),
    )

    yield (
        'ows:Constraint',
        {'name': 'QueryExpressions'},
        (
            'ows:AllowedValues',
            tpl.ows_value('wfs:Query'),
        ),
    )

    if ta.service.withInspireMeta:
        yield 'ows:ExtendedCapabilities/inspire_dls:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)


def operation_params(ta, op):
    versions = [tpl.ows_value(v) for v in ta.service.supportedVersions]
    formats = [tpl.ows_value(f) for f in op.formats]

    if op.verb == gws.OwsVerb.GetCapabilities:
        yield 'ows:Parameter', {'name': 'acceptVersions'}, ('ows:AllowedValues', versions)
        yield 'ows:Parameter', {'name': 'acceptFormats'}, ('ows:AllowedValues', formats)
    if op.verb == gws.OwsVerb.DescribeFeatureType:
        yield 'ows:Parameter', {'name': 'outputFormat'}, ('ows:AllowedValues', formats)
    if op.verb == gws.OwsVerb.GetFeature:
        yield 'ows:Parameter', {'name': 'outputFormat'}, ('ows:AllowedValues', formats)
        yield (
            'ows:Parameter',
            {'name': 'resultType'},
            (
                'ows:AllowedValues',
                tpl.ows_value('results'),
                tpl.ows_value('hits'),
            ),
        )


def feature_type_list(ta: server.TemplateArgs):
    seen = set()
    for lc in ta.layerCapsList:
        if lc.featureNameQ in seen:
            continue
        seen.add(lc.featureNameQ)
        yield ['FeatureType', feature_type(ta, lc)]


def feature_type(ta: server.TemplateArgs, lc: server.LayerCaps):
    yield 'Name', lc.featureNameQ
    yield 'Title', lc.layer.title
    yield 'Abstract', lc.layer.metadata.abstract

    for n, b in enumerate(lc.bounds):
        if n == 0:
            yield 'DefaultCRS', b.crs.urn
        else:
            yield 'OtherCRS', b.crs.urn

    yield tpl.ows_wgs84_bounding_box(lc)
    yield tpl.meta_links_simple(ta, lc.layer.metadata)


def filters(ta: server.TemplateArgs):
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
    return ns + ':Constraint', {'name': n}, ['ows:NoValues'], ['ows:DefaultValue', value]
