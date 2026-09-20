"""WFS GetCapabilities template."""

import gws
import gws.lib.xmlx
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
from gws.lib.xmlx import tag


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        tag(
            'WFS:WFS_Capabilities',
            {'version': ta.version},
            doc(ta),
        ),
        default_namespace=gws.lib.xmlx.namespace.c.WFS,
        extra_namespaces=[gws.lib.xmlx.namespace.c.GML, *tpl.namespaces_from_caps(ta)],
    )


def doc(ta: server.TemplateArgs):
    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)
    yield tag('OWS_11:OperationsMetadata', operations(ta))
    yield tag('WFS:FeatureTypeList', feature_type_list(ta))
    yield tag('FES:Filter_Capabilities', filters(ta))


def operations(ta: server.TemplateArgs):
    versions = [tpl.ows_value(v) for v in ta.service.supportedVersions]

    for op in ta.service.supportedOperations:
        yield tag(
            'OWS_11:Operation',
            {'name': op.verb},
            tpl.ows_service_url(ta),
            operation_params(ta, op),
        )

    yield tag('OWS_11:Parameter', {'name': 'version'}, tag('OWS_11:AllowedValues', versions))

    yield constraint('OWS_11', 'ImplementsBasicWFS', 'TRUE')
    yield constraint('OWS_11', 'KVPEncoding', 'TRUE')
    yield constraint('OWS_11', 'ImplementsTransactionalWFS', 'FALSE')
    yield constraint('OWS_11', 'ImplementsLockingWFS', 'FALSE')
    yield constraint('OWS_11', 'XMLEncoding', 'FALSE')
    yield constraint('OWS_11', 'SOAPEncoding', 'FALSE')
    yield constraint('OWS_11', 'ImplementsInheritance', 'FALSE')
    yield constraint('OWS_11', 'ImplementsRemoteResolve', 'FALSE')
    yield constraint('OWS_11', 'ImplementsResultPaging', 'TRUE')
    yield constraint('OWS_11', 'ImplementsStandardJoins', 'FALSE')
    yield constraint('OWS_11', 'ImplementsSpatialJoins', 'FALSE')
    yield constraint('OWS_11', 'ImplementsTemporalJoins', 'FALSE')
    yield constraint('OWS_11', 'ImplementsFeatureVersioning', 'FALSE')
    yield constraint('OWS_11', 'ManageStoredQueries', 'FALSE')
    yield constraint('OWS_11', 'CountDefault', ta.service.maxFeatureCount)

    yield tag(
        'OWS_11:Constraint',
        {'name': 'QueryExpressions'},
        tag(
            'OWS_11:AllowedValues',
            tpl.ows_value('wfs:Query'),
        ),
    )

    if ta.service.withInspireMeta:
        md = ta.service.metadata
        yield tag(
            'OWS_11:ExtendedCapabilities/INSPIRE_DLS:ExtendedCapabilities',
            tpl.inspire_extended_capabilities(ta),
            tag(
                'INSPIRE_DLS:SpatialDataSetIdentifier',
                {'metadataURL': md.metaLinks[0].url if md.metaLinks else ''},
                tag('INSPIRE_COMMON:Code', md.catalogUid),
            ),
        )


def operation_params(ta, op):
    versions = [tpl.ows_value(v) for v in ta.service.supportedVersions]
    formats = [tpl.ows_value(f) for f in op.formats]

    if op.verb == gws.OwsVerb.GetCapabilities:
        yield tag('OWS_11:Parameter', {'name': 'acceptVersions'}, tag('OWS_11:AllowedValues', versions))
        yield tag('OWS_11:Parameter', {'name': 'acceptFormats'}, tag('OWS_11:AllowedValues', formats))

    if op.verb == gws.OwsVerb.DescribeFeatureType:
        yield tag('OWS_11:Parameter', {'name': 'outputFormat'}, tag('OWS_11:AllowedValues', formats))

    if op.verb == gws.OwsVerb.GetFeature:
        yield tag('OWS_11:Parameter', {'name': 'outputFormat'}, tag('OWS_11:AllowedValues', formats))
        yield tag(
            'OWS_11:Parameter',
            {'name': 'resultType'},
            tag(
                'OWS_11:AllowedValues',
                tpl.ows_value('results'),
                tpl.ows_value('hits'),
            ),
        )


def feature_type_list(ta: server.TemplateArgs):
    seen = set()
    for lc in ta.layerCapsList:
        name = tpl.feature_name(ta, lc)
        if name in seen:
            continue
        seen.add(name)
        yield tag('WFS:FeatureType', feature_type(ta, lc))


def feature_type(ta: server.TemplateArgs, lc: server.LayerCaps):
    yield tag('WFS:Name', tpl.feature_name(ta, lc))
    yield tag('WFS:Title', lc.layer.title)
    yield tag('WFS:Abstract', lc.layer.metadata.abstract)

    for n, b in enumerate(lc.bounds):
        if n == 0:
            yield tag('WFS:DefaultCRS', b.crs.urn)
        else:
            yield tag('WFS:OtherCRS', b.crs.urn)

    yield tpl.ows_wgs84_bounding_box(lc)
    yield tpl.meta_links_simple(ta, lc.layer.metadata)


def filters(ta: server.TemplateArgs):
    yield tag(
        'FES:Conformance',
        constraint('FES', 'ImplementsAdHocQuery', 'TRUE'),
        constraint('FES', 'ImplementsMinSpatialFilter', 'TRUE'),
        constraint('FES', 'ImplementsQuery', 'TRUE'),
        constraint('FES', 'ImplementsResourceId', 'TRUE'),
        constraint('FES', 'ImplementsMinStandardFilter', 'TRUE'),
        constraint('FES', 'ImplementsMinTemporalFilter', 'TRUE'),
        constraint('FES', 'ImplementsExtendedOperators', 'FALSE'),
        constraint('FES', 'ImplementsFunctions', 'FALSE'),
        constraint('FES', 'ImplementsMinimumXPath', 'FALSE'),
        constraint('FES', 'ImplementsSorting', 'FALSE'),
        constraint('FES', 'ImplementsSpatialFilter', 'FALSE'),
        constraint('FES', 'ImplementsStandardFilter', 'FALSE'),
        constraint('FES', 'ImplementsTemporalFilter', 'FALSE'),
        constraint('FES', 'ImplementsVersionNav', 'FALSE'),
    )

    yield tag('FES:Id_Capabilities/FES:ResourceIdentifier', {'name': 'fes:ResourceId'})

    yield tag(
        'FES:Spatial_Capabilities',
        tag('FES:GeometryOperands/FES:GeometryOperand', {'name': 'gml:Envelope'}),
        tag('FES:SpatialOperators/FES:SpatialOperator', {'name': 'BBOX'}),
    )


def constraint(ns, name, value):
    return tag(f'{ns}:Constraint', {'name': name}, tag('OWS_11:NoValues'), tag('OWS_11:DefaultValue', value))
