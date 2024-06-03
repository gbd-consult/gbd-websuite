"""Utilities to deal with LayerCaps objects."""

from typing import Optional, Callable

import gws
import gws.lib.xmlx as xmlx
import gws.lib.uom
import gws.gis.extent

from . import core


def for_layer(layer: gws.Layer, user: gws.User, service: Optional[gws.OwsService] = None) -> core.LayerCaps:
    lc = core.LayerCaps(leaves=[], children=[])

    lc.layer = layer
    lc.title = layer.title
    lc.model = layer.root.app.modelMgr.find_model(layer, user=user, access=gws.Access.read)

    opts = layer.owsOptions

    geom_name = opts.geometryName
    if not geom_name and lc.model:
        geom_name = lc.model.geometryName
    if not geom_name:
        geom_name = 'geometry'

    lc.layerName = xmlx.namespace.unqualify_name(opts.layerName)
    lc.featureName = xmlx.namespace.unqualify_name(opts.featureName)
    lc.geometryName = xmlx.namespace.unqualify_name(geom_name)

    lc.xmlNamespace = opts.xmlNamespace

    if lc.xmlNamespace:
        lc.layerQname = xmlx.namespace.qualify_name(lc.layerName, lc.xmlNamespace)
        lc.featureQname = xmlx.namespace.qualify_name(lc.featureName, lc.xmlNamespace)
        lc.geometryQname = xmlx.namespace.qualify_name(lc.geometryName, lc.xmlNamespace)
    else:
        lc.layerQname = lc.layerName
        lc.featureQname = lc.featureName
        lc.geometryQname = lc.geometryName

    lc.hasLegend = layer.hasLegend
    lc.hasSearch = layer.isSearchable

    scales = [gws.lib.uom.res_to_scale(r) for r in layer.resolutions]
    lc.minScale = int(min(scales))
    lc.maxScale = int(max(scales))

    lc.bounds = []
    if service:
        lc.bounds = [
            gws.Bounds(
                crs=b.crs,
                extent=gws.gis.extent.transform_from_wgs(layer.wgsExtent, b.crs)
            )
            for b in service.supportedBounds
        ]

    return lc


def layer_name_matches(lc: core.LayerCaps, name: str) -> bool:
    if ':' in name:
        return name == lc.layerQname
    else:
        return name == lc.layerName


def feature_name_matches(lc: core.LayerCaps, name: str) -> bool:
    if ':' in name:
        return name == lc.featureQname
    else:
        return name == lc.featureName


def xml_schema(lcs: list[core.LayerCaps], user: gws.User) -> gws.XmlElement:
    ns = None

    for lc in lcs:
        if not lc.xmlNamespace:
            gws.log.debug(f'xml_schema: skip {lc.layer.uid}: no xmlns')
            continue
        if not ns:
            ns = lc.xmlNamespace
            continue
        if lc.xmlNamespace.xmlns != ns.xmlns:
            gws.log.debug(f'xml_schema: skip {lc.layer.uid}: wrong xmlns')
            continue

    if not ns:
        raise gws.NotFoundError('xml_schema: no xmlns found')

    tag = [
        'xsd:schema',
        {
            f'xmlns:{ns.xmlns}': ns.uri,
            'targetNamespace': ns.uri,
            'elementFormDefault': 'qualified',
        }
    ]

    if ns.extendsGml:
        gml = xmlx.namespace.get('gml3')
        tag.append(['xsd:import', {'namespace': gml.uri, 'schemaLocation': gml.schemaLocation}])

    for lc in lcs:
        elements = []
        for f in lc.model.fields:
            if user.can_read(f):
                typ = _ATTR_TO_XSD.get(f.attributeType)
                if typ:
                    elements.append(['xsd:element', {
                        'maxOccurs': '1',
                        'minOccurs': '0',
                        'nillable': 'false' if f.isRequired else 'true',
                        'name': f.name,
                        'type': typ,
                    }])

        type_name = f'{lc.featureName}Type'

        type_def = ['xsd:complexContent']
        if ns.extendsGml:
            type_def.append(['xsd:extension', {'base': 'gml:AbstractFeatureType'}])
        type_def.append(['xsd:sequence', elements])

        atts = {
            'name': lc.featureQname,
            'type': type_name,
        }
        if ns.extendsGml:
            atts['substitutionGroup'] = 'gml:AbstractFeature'

        tag.append(['xsd:complexType', {'name': type_name}, type_def])
        tag.append(['xsd:element', atts])

    return xmlx.tag(*tag)


# map attributes types to XSD
# https://www.w3.org/TR/xmlschema11-2/#built-in-primitive-datatypes

_ATTR_TO_XSD = {
    gws.AttributeType.bool: 'xsd:boolean',
    gws.AttributeType.bytes: 'xsd:hexBinary',
    gws.AttributeType.date: 'xsd:date',
    gws.AttributeType.datetime: 'xsd:dateTime',
    gws.AttributeType.feature: '',
    gws.AttributeType.featurelist: '',
    gws.AttributeType.file: '',
    gws.AttributeType.float: 'xsd:float',
    gws.AttributeType.floatlist: '',
    gws.AttributeType.geometry: 'gml:GeometryPropertyType',
    gws.AttributeType.int: 'xsd:decimal',
    gws.AttributeType.intlist: '',
    gws.AttributeType.str: 'xsd:string',
    gws.AttributeType.strlist: '',
    gws.AttributeType.time: 'xsd:time',
}
