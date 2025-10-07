"""Utilities to deal with LayerCaps objects."""

from typing import Optional

import gws
import gws.lib.extent
import gws.lib.uom
import gws.lib.xmlx as xmlx

from . import core


def for_layer(layer: gws.Layer, user: gws.User, service: Optional[gws.OwsService] = None) -> core.LayerCaps:
    """Create ``LayerCaps`` for a layer."""

    lc = core.LayerCaps(leaves=[], children=[])

    lc.layer = layer
    lc.title = layer.title
    lc.model = layer.root.app.modelMgr.find_model(layer.ows, layer, user=user, access=gws.Access.read)

    geom_name = layer.ows.geometryName
    if not geom_name and lc.model:
        geom_name = lc.model.geometryName
    if not geom_name:
        geom_name = 'geometry'

    lc.layerName = layer.ows.layerName
    lc.featureName = layer.ows.featureName
    lc.geometryName = geom_name

    lc.xmlNamespace = layer.ows.xmlNamespace

    if lc.xmlNamespace:
        lc.layerNameQ = xmlx.namespace.qualify_name(lc.layerName, lc.xmlNamespace)
        lc.featureNameQ = xmlx.namespace.qualify_name(lc.featureName, lc.xmlNamespace)
        lc.geometryNameQ = xmlx.namespace.qualify_name(lc.geometryName, lc.xmlNamespace)
    else:
        lc.layerNameQ = lc.layerName
        lc.featureNameQ = lc.featureName
        lc.geometryNameQ = lc.geometryName

    lc.hasLegend = layer.hasLegend
    lc.isSearchable = layer.isSearchable

    scales = [gws.lib.uom.res_to_scale(r) for r in layer.resolutions]
    lc.minScale = int(min(scales))
    lc.maxScale = int(max(scales))

    lc.bounds = []
    if service:
        lc.bounds = [
            gws.Bounds(
                crs=b.crs,
                extent=gws.lib.extent.transform_from_wgs(layer.wgsExtent, b.crs),
            )
            for b in service.supportedBounds
        ]

    return lc


def layer_name_matches(lc: core.LayerCaps, name: str) -> bool:
    """Check if the layer name in the caps matches the given name."""

    if ':' in name:
        return name == lc.layerNameQ
    else:
        return name == lc.layerName


def feature_name_matches(lc: core.LayerCaps, name: str, xmlns_replacements: dict) -> bool:
    """Check if the feature name in the caps matches the given name."""

    if name == lc.featureNameQ:
        return True

    if ':' not in name:
        return name == lc.featureName

    custom_xmlns, _, name = xmlx.namespace.split_name(name)
    if name == lc.featureName and lc.xmlNamespace and lc.xmlNamespace.uid in xmlns_replacements:
        return xmlns_replacements[lc.xmlNamespace.uid] == custom_xmlns

    return False


def xml_schema(lcs: list[core.LayerCaps], user: gws.User) -> tuple[gws.XmlElement, gws.XmlOptions]:
    """Create an ad-hoc XML Schema for a list of `LayerCaps`."""

    ns = None

    for lc in lcs:
        if not lc.xmlNamespace:
            raise gws.NotFoundError(f'xml_schema: {lc.layer.uid}: no xmlns')
        if not lc.model:
            raise gws.NotFoundError(f'xml_schema: {lc.layer.uid}: no model')
        if not ns:
            ns = lc.xmlNamespace
        elif lc.xmlNamespace.xmlns != ns.xmlns:
            raise gws.NotFoundError(f'xml_schema: {lc.layer.uid}: wrong xmlns: {ns.xmlns=} {lc.xmlNamespace.xmlns=}')

    if not ns:
        raise gws.NotFoundError('xml_schema: no xmlns found')

    opts = gws.XmlOptions()
    opts.namespaces = {}
    opts.namespaces[ns.xmlns] = ns

    tag = [
        'xsd:schema',
        {
            'targetNamespace': ns.uri,
            'elementFormDefault': 'qualified',
        },
    ]

    if ns.extendsGml:
        gml = xmlx.namespace.require('gml')
        opts.namespaces[gml.xmlns] = gml
        tag.append(['xsd:import', {'namespace': gml.uri, 'schemaLocation': gml.schemaLocation}])

    seen = set()
    
    for lc in lcs:
        if lc.featureName in seen:
            continue
        seen.add(lc.featureName)
        
        elements = []

        for f in gws.u.require(lc.model).fields:
            if user.can_read(f):
                elements.append(
                    [
                        'xsd:element',
                        {
                            'maxOccurs': '1',
                            'minOccurs': '0',
                            'nillable': 'false' if f.isRequired else 'true',
                            'name': f.name,
                            'type': _xsd_type(f),
                        },
                    ]
                )

        type_name = f'{lc.featureName}Type'

        type_def = []
        type_def.append('xsd:complexContent')
        if ns.extendsGml:
            type_def.append(
                ['xsd:extension', {'base': 'gml:AbstractFeatureType'}, ['xsd:sequence', elements]],
            )
        else:
            type_def.append(['xsd:sequence', elements])

        tag.append(['xsd:complexType', {'name': type_name}, type_def])

        atts = {
            'name': lc.featureName,
            'type': ns.xmlns + ':' + type_name,
        }
        if ns.extendsGml:
            atts['substitutionGroup'] = 'gml:AbstractFeature'

        tag.append(['xsd:element', atts])

    return xmlx.tag(*tag), opts


def _xsd_type(f: gws.ModelField) -> str:
    """Get the XSD type for a model field."""

    if f.attributeType != gws.AttributeType.geometry:
        return _ATTR_TO_XSD.get(f.attributeType, 'xsd:string')

    typ = gws.u.get(f, 'geometryType')
    if not typ:
        return 'gml:GeometryPropertyType'
    return _GEOM_TO_XSD.get(typ, 'gml:GeometryPropertyType')


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

_GEOM_TO_XSD = {
    gws.GeometryType.point: 'gml:PointPropertyType',
    gws.GeometryType.linestring: 'gml:CurvePropertyType',
    gws.GeometryType.polygon: 'gml:SurfacePropertyType',
    gws.GeometryType.multipoint: 'gml:MultiPointPropertyType',
    gws.GeometryType.multilinestring: 'gml:MultiCurvePropertyType',
    gws.GeometryType.multipolygon: 'gml:MultiSurfacePropertyType',
    gws.GeometryType.multicurve: 'gml:MultiCurvePropertyType',
    gws.GeometryType.multisurface: 'gml:MultiSurfacePropertyType',
    gws.GeometryType.linearring: 'gml:LinearRingPropertyType',
    gws.GeometryType.tin: 'gml:TriangulatedSurfacePropertyType',
    gws.GeometryType.surface: 'gml:SurfacePropertyType',
}
