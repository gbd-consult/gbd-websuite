"""CSW GetCapabilities template (ISO)."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        ('csw:Capabilities', {'version': ta.version}, caps(ta)),
        extra_namespaces=[gws.lib.xmlx.namespace.get('gml')]
    )


def caps(ta: server.TemplateArgs):
    yield tpl.ows_service_identification(ta)
    yield tpl.ows_service_provider(ta)

    yield (
        'ows:OperationsMetadata',
        (
            'ows:Operation',
            {'name': 'GetCapabilities'},
            tpl.ows_service_url(ta),
            ('ows:Parameter', {'name': 'sections'},
             ('ows:Value', 'ServiceIdentification'),
             ('ows:Value', 'ServiceProvider'),
             ('ows:Value', 'OperationsMetadata'),
             ('ows:Value', 'Filter_Capabilities'),
             )
        ),
        (
            'ows:Operation',
            {'name': 'DescribeRecord'},
            tpl.ows_service_url(ta),
            ('ows:Parameter', {'name': 'typeName'}, ('ows:Value', 'gmd:MD_Metadata')),
            ('ows:Parameter', {'name': 'outputFormat'}, ('ows:Value', 'application/xml')),
            ('ows:Parameter', {'name': 'schemaLanguage'}, ('ows:Value', 'http://www.w3.org/XML/Schema')),
            ('ows:Parameter', {'name': 'resultType'}, ('ows:Value', 'hits'), ('ows:Value', 'results')),
            ('ows:Parameter', {'name': 'ElementSetName'}, ('ows:Value', 'full')),
            ('ows:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, ('ows:Value', 'FILTER')),
            ('ows:Parameter', {'name': 'version'}, ('ows:Value', ta.version))
        ),
        (
            'ows:Operation',
            {'name': 'GetRecords'},
            tpl.ows_service_url(ta, post=True),
            ('ows:Parameter', {'name': 'typeName'}, ('ows:Value', 'csw:Record')),
            ('ows:Parameter', {'name': 'outputFormat'}, ('ows:Value', 'application/xml')),
            ('ows:Parameter', {'name': 'outputSchema'}, ('ows:Value', 'http://www.opengis.net/cat/csw/2.0.2')),
            ('ows:Parameter', {'name': 'resultType'}, ('ows:Value', 'results')),
            ('ows:Parameter', {'name': 'ElementSetName'}, ('ows:Value', 'full')),
            ('ows:Parameter', {'name': 'CONSTRAINTLANGUAGE'}, ('ows:Value', 'FILTER')),
            ('ows:Parameter', {'name': 'version'}, ('ows:Value', ta.version))
        ),

        ('ows:Constraint', {'name': 'IsoProfiles'}, ('ows:Value', 'http://www.isotc211.org/2005/gmd')),
        ('inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta))
    )

    yield (
        'ogc:Filter_Capabilities',
        (
            'ogc:Spatial_Capabilities',
            ('ogc:GeometryOperands/ogc:GeometryOperand', 'gml:Envelope'),
            ('ogc:SpatialOperators/ogc:SpatialOperator', {'name': 'BBOX'})
        ),
        (
            'ogc:Scalar_Capabilities',
            ('ogc:LogicalOperators', ''),
            (
                'ogc:ComparisonOperators',
                ('ogc:ComparisonOperator', 'EqualTo'),
                ('ogc:ComparisonOperator', 'NotEqualTo'),
                ('ogc:ComparisonOperator', 'NullCheck'),
            )
        ),
        (
            'ogc:Id_Capabilities',
            ('ogc:EID',),
            ('ogc:FID',),
        ),
    )
