"""CSW 2.0.2 GetCapabilities template (ISO)."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx
from gws.lib.xmlx import tag


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        tag('csw:Capabilities', {'version': ta.version}, caps(ta)),
        namespaces={
            'ows': gws.lib.xmlx.namespace.require('ows0'),
            'gml': gws.lib.xmlx.namespace.require('gml0'),
        },
    )


def caps(ta: server.TemplateArgs):
    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)

    yield tag(
        'ows:OperationsMetadata',
        tag(
            'ows:Operation',
            {'name': 'GetCapabilities'},
            tpl.ows_service_url(ta),
            tag(
                'ows:Parameter',
                {'name': 'sections'},
                tag('ows:Value', 'ServiceIdentification'),
                tag('ows:Value', 'ServiceProvider'),
                tag('ows:Value', 'OperationsMetadata'),
                tag('ows:Value', 'Filter_Capabilities'),
            ),
        ),
        tag(
            'ows:Operation',
            {'name': 'DescribeRecord'},
            tpl.ows_service_url(ta),
            tag('ows:Parameter', {'name': 'typeName'}, tag('ows:Value', 'gmd:MD_Metadata')),
            tag('ows:Parameter', {'name': 'outputFormat'}, tag('ows:Value', 'application/xml')),
            tag('ows:Parameter', {'name': 'schemaLanguage'}, tag('ows:Value', 'http://www.w3.org/XML/Schema')),
            tag('ows:Parameter', {'name': 'resultType'}, tag('ows:Value', 'hits'), tag('ows:Value', 'results')),
            tag('ows:Parameter', {'name': 'ElementSetName'}, tag('ows:Value', 'full')),
            tag('ows:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, tag('ows:Value', 'FILTER')),
            tag('ows:Parameter', {'name': 'version'}, tag('ows:Value', ta.version)),
        ),
        tag(
            'ows:Operation',
            {'name': 'GetRecords'},
            tpl.ows_service_url(ta, post=True),
            tag('ows:Parameter', {'name': 'typeName'}, tag('ows:Value', 'gmd:MD_Metadata')),
            tag('ows:Parameter', {'name': 'outputFormat'}, tag('ows:Value', 'application/xml')),
            tag('ows:Parameter', {'name': 'outputSchema'}, tag('ows:Value', 'http://www.opengis.net/cat/csw/2.0.2')),
            tag('ows:Parameter', {'name': 'resultType'}, tag('ows:Value', 'results')),
            tag('ows:Parameter', {'name': 'ElementSetName'}, tag('ows:Value', 'full')),
            tag('ows:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, tag('ows:Value', 'FILTER')),
            tag('ows:Parameter', {'name': 'version'}, tag('ows:Value', ta.version)),
        ),
        tag(
            'ows:Constraint',
            {'name': 'IsoProfiles'},
            tag('ows:Value', 'http://www.isotc211.org/2005/gmd'),
        ),
        tag('ows:ExtendedCapabilities/inspire_ds:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)),
    )

    yield tag(
        'ogc:Filter_Capabilities',
        tag(
            'ogc:Spatial_Capabilities',
            tag('ogc:GeometryOperands/ogc:GeometryOperand', 'gml:Envelope'),
            tag('ogc:SpatialOperators/ogc:SpatialOperator', {'name': 'BBOX'}),
        ),
        tag(
            'ogc:Scalar_Capabilities',
            tag('ogc:LogicalOperators', ''),
            tag(
                'ogc:ComparisonOperators',
                tag('ogc:ComparisonOperator', 'EqualTo'),
                tag('ogc:ComparisonOperator', 'NotEqualTo'),
                tag('ogc:ComparisonOperator', 'NullCheck'),
            ),
        ),
        tag(
            'ogc:Id_Capabilities',
            tag('ogc:EID',),
            tag('ogc:FID',),
        ),
    )
