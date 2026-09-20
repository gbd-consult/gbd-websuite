"""CSW 2.0.2 GetCapabilities template (ISO)."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx
from gws.lib.xmlx import tag

def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        tag('CSW:Capabilities', {'version': ta.version}, caps(ta)),
        extra_namespaces=[gws.lib.xmlx.namespace.c.GML_3_1, gws.lib.xmlx.namespace.c.GMD],
    )


def caps(ta: server.TemplateArgs):
    yield tpl.ows_service_identification(ta, ns='OWS_0')
    yield tpl.ows_service_provider(ta, ns='OWS_0')

    yield tag(
        'OWS_0:OperationsMetadata',
        tag(
            'OWS_0:Operation',
            {'name': 'GetCapabilities'},
            tpl.ows_service_url(ta, ns='OWS_0'),
            tag(
                'OWS_0:Parameter',
                {'name': 'sections'},
                tag('OWS_0:Value', 'ServiceIdentification'),
                tag('OWS_0:Value', 'ServiceProvider'),
                tag('OWS_0:Value', 'OperationsMetadata'),
                tag('OWS_0:Value', 'Filter_Capabilities'),
            ),
        ),
        tag(
            'OWS_0:Operation',
            {'name': 'DescribeRecord'},
            tpl.ows_service_url(ta, ns='OWS_0'),
            tag('OWS_0:Parameter', {'name': 'typeName'}, tag('OWS_0:Value', 'gmd:MD_Metadata')),
            tag('OWS_0:Parameter', {'name': 'outputFormat'}, tag('OWS_0:Value', 'application/xml')),
            tag('OWS_0:Parameter', {'name': 'schemaLanguage'}, tag('OWS_0:Value', 'http://www.w3.org/XML/Schema')),
            tag('OWS_0:Parameter', {'name': 'resultType'}, tag('OWS_0:Value', 'hits'), tag('OWS_0:Value', 'results')),
            tag('OWS_0:Parameter', {'name': 'ElementSetName'}, tag('OWS_0:Value', 'full')),
            tag('OWS_0:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, tag('OWS_0:Value', 'FILTER')),
            tag('OWS_0:Parameter', {'name': 'version'}, tag('OWS_0:Value', ta.version)),
        ),
        tag(
            'OWS_0:Operation',
            {'name': 'GetRecords'},
            tpl.ows_service_url(ta, post=True, ns='OWS_0'),
            tag('OWS_0:Parameter', {'name': 'typeName'}, tag('OWS_0:Value', 'gmd:MD_Metadata')),
            tag('OWS_0:Parameter', {'name': 'outputFormat'}, tag('OWS_0:Value', 'application/xml')),
            tag('OWS_0:Parameter', {'name': 'outputSchema'}, tag('OWS_0:Value', 'http://www.opengis.net/cat/csw/2.0.2')),
            tag('OWS_0:Parameter', {'name': 'resultType'}, tag('OWS_0:Value', 'results')),
            tag('OWS_0:Parameter', {'name': 'ElementSetName'}, tag('OWS_0:Value', 'full')),
            tag('OWS_0:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, tag('OWS_0:Value', 'FILTER')),
            tag('OWS_0:Parameter', {'name': 'version'}, tag('OWS_0:Value', ta.version)),
        ),
        tag(
            'OWS_0:Constraint',
            {'name': 'IsoProfiles'},
            tag('OWS_0:Value', 'http://www.isotc211.org/2005/gmd'),
        ),
        tag('OWS_0:ExtendedCapabilities/INSPIRE_DS:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)),
    )

    yield tag(
        'OGC:Filter_Capabilities',
        tag(
            'OGC:Spatial_Capabilities',
            tag('OGC:GeometryOperands/OGC:GeometryOperand', 'gml:Envelope'),
            tag('OGC:SpatialOperators/OGC:SpatialOperator', {'name': 'BBOX'}),
        ),
        tag(
            'OGC:Scalar_Capabilities',
            tag('OGC:LogicalOperators', ''),
            tag(
                'OGC:ComparisonOperators',
                tag('OGC:ComparisonOperator', 'EqualTo'),
                tag('OGC:ComparisonOperator', 'NotEqualTo'),
                tag('OGC:ComparisonOperator', 'NullCheck'),
            ),
        ),
        tag(
            'OGC:Id_Capabilities',
            tag('OGC:EID',),
            tag('OGC:FID',),
        ),
    )
