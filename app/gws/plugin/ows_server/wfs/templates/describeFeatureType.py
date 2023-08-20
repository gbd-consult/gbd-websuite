import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx


def main(args: dict):
    ta = args.get('owsArgs')
    return tpl.to_xml(args, ('xsd:schema', schema(ta)))


def schema(ta: tpl.TemplateArgs):
    lc = ta.layerCapsList[0]
    ns = lc.layer.owsOptions.xmlNamespace

    yield {
        'xmlns:xsd': '',
        'xmlns:gml': '',
        'xmlns:' + ns.xmlns: '',
        'targetNamespace': ns.uri,
        'elementFormDefault': 'qualified',
    }

    gml = xmlx.namespace.get('gml')

    yield 'xsd:import', {
        'namespace': gml.uri,
        'schemaLocation': gml.schemaLocation,
    }

    type_name = lc.featureName + 'Type'

    yield 'xsd:complexType', {'name': type_name}, (
        'xsd:complexContent',
        ('xsd:extension', {'base': 'gml:AbstractFeatureType'}),
        ('xsd:sequence', elements(ta, lc))
    )

    yield 'xsd:element', {
        'name': lc.featureQname,
        'substitutionGroup': 'gml:AbstractFeature',
        'type': type_name
    }


def elements(ta: tpl.TemplateArgs, lc: server.LayerCaps):
    for f in lc.model.fields:
        yield 'xsd:element', {
            'maxOccurs': '1',
            'minOccurs': '0',
            'nillable': 'true',
            'name': f.name,
            'type': f.type,

        }
