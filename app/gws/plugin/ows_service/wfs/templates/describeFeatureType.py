import gws.lib.xmlx as xmlx
import gws.plugin.ows_service.templatelib as tpl


def main(ARGS):
    def elements(lc):
        for name, typ in lc.adhoc_feature_schema.items():
            yield 'xsd:element', {
                'maxOccurs': '1',
                'minOccurs': '0',
                'nillable': 'true',
                'name': name,
                'type': typ,

            }

    def adhoc_feature_schema(lc):
        type_name = lc.feature_pname + 'Type'
        yield (
            'xsd:complexType', {'name': type_name},
            (
                'xsd:complexContent',
                ('xsd:extension', {'base': 'gml:AbstractFeatureType'}),
                ('xsd:sequence', elements(lc))
            )
        )

        yield 'xsd:element', {
            'name': lc.feature_qname,
            'substitutionGroup': 'gml:AbstractFeature',
            'type': type_name
        }

    def schema():
        pfx, _ = xmlx.split_name(ARGS.layer_caps_list[0].feature_qname)
        yield {
            'xmlns:xsd': '',
            'xmlns:gml': '',
            'xmlns:' + pfx: '',
            'targetNamespace': xmlx.namespaces.uri(pfx),
            'elementFormDefault': 'qualified',
        }

        yield (
            'xsd:import', {
                'namespace': xmlx.namespaces.uri('gml'),
                'schemaLocation': xmlx.namespaces.schema('gml')
            })
        for lc in ARGS.layer_caps_list:
            if lc.adhoc_feature_schema:
                yield adhoc_feature_schema(lc)

    ##

    return tpl.to_xml(ARGS, ('xsd:schema', schema()))
