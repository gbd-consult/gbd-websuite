"""Utilities for LayerCaps objects."""

from typing import Optional

import gws
import gws.lib.extent
import gws.gis.zoom
import gws.lib.xmlx as xmlx
from gws.lib.xmlx import tag

from . import core


def for_layer(layer: gws.Layer, user: gws.User, service: Optional[gws.OwsService] = None) -> core.LayerCaps:
    """Create ``LayerCaps`` for a layer.

    ``isGroup`` is not set and ``children`` and ``leaves`` are empty;
    the caller fills them in when building the layer tree.

    Args:
        layer: Layer object.
        user: User, used to find a readable model for the layer.
        service: Service; if given, ``bounds`` are computed for its supported bounds.

    Returns:
        The layer caps.
    """

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

    lc.hasLegend = layer.hasLegend
    lc.isSearchable = layer.isSearchable

    scales = [gws.gis.zoom.res_to_scale(r, layer.mapCrs) for r in layer.resolutions]
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
    """Check if the layer name in the caps matches the given name, ignoring a prefix.

    Args:
        lc: Layer caps.
        name: Layer name, possibly with a namespace prefix.

    Returns:
        ``True`` if the names match.
    """

    return xmlx.namespace.plain_name(name) == lc.layerName

def feature_prefix(lc: core.LayerCaps, custom_namespace_prefixes: dict) -> str:
    """Return the namespace prefix of the feature type.

    Args:
        lc: Layer caps.
        custom_namespace_prefixes: Custom prefixes by namespace uri, as requested with the WFS ``NAMESPACES`` parameter.

    Returns:
        The custom prefix for the layer namespace if there is one, otherwise the namespace prefix,
        or an empty string if the layer has no namespace.
    """

    ns = lc.xmlNamespace
    if not ns:
        return ''
    return custom_namespace_prefixes.get(ns.uri, ns.prefix)


def qualified_feature_name(lc: core.LayerCaps, custom_namespace_prefixes: dict) -> str:
    """Return the feature type name as a QName (``prefix:name``), for use in text content.

    Args:
        lc: Layer caps.
        custom_namespace_prefixes: Custom prefixes by namespace uri.

    Returns:
        The qualified name, or the plain feature name if the layer has no namespace.
    """

    prefix = feature_prefix(lc, custom_namespace_prefixes)
    return prefix + ':' + lc.featureName if prefix else lc.featureName


def feature_name_matches(lc: core.LayerCaps, name: str, custom_namespace_prefixes: dict) -> bool:
    """Check if the feature name in the caps matches the given name, which may be a QName.

    A name without a prefix matches any namespace.

    Args:
        lc: Layer caps.
        name: Feature name, optionally with a prefix.
        custom_namespace_prefixes: Custom prefixes by namespace uri.

    Returns:
        ``True`` if the names match.
    """

    _, prefix, pname = xmlx.namespace.parse_name(name)
    if pname != lc.featureName:
        return False
    return not prefix or prefix == feature_prefix(lc, custom_namespace_prefixes)


def xml_schema(lcs: list[core.LayerCaps], user: gws.User) -> tuple[gws.XmlElement, gws.XmlOptions]:
    """Create an ad-hoc XML Schema for a list of ``LayerCaps``.

    All layers must have a model and share the same XML namespace. The schema
    contains a complex type and an element for each feature type, with elements
    for the model fields the user can read.

    Args:
        lcs: Layer caps.
        user: User.

    Returns:
        The schema element and XML options to serialize it.

    Raises:
        ``gws.NotFoundError``: If a layer has no namespace or model, or the namespaces differ.
    """

    ns = None

    for lc in lcs:
        if not lc.xmlNamespace:
            raise gws.NotFoundError(f'xml_schema: {lc.layer.uid}: no xmlns')
        if not lc.model:
            raise gws.NotFoundError(f'xml_schema: {lc.layer.uid}: no model')
        if not ns:
            ns = lc.xmlNamespace
        elif lc.xmlNamespace.prefix != ns.prefix:
            raise gws.NotFoundError(f'xml_schema: {lc.layer.uid}: wrong xmlns: {ns.prefix=} {lc.xmlNamespace.prefix=}')

    if not ns:
        raise gws.NotFoundError('xml_schema: no xmlns found')

    opts = gws.XmlOptions()

    schema = tag(
        'XSD:schema',
        {
            'targetNamespace': ns.uri,
            'elementFormDefault': 'qualified',
        },
    )
    schema.declare(ns)

    if ns.extendsGml:
        gml = xmlx.namespace.c.GML
        schema.declare(gml)
        schema.append(tag('XSD:import', {'namespace': gml.uri, 'schemaLocation': gml.schemaLocation}))

    seen = set()

    for lc in lcs:
        if lc.featureName in seen:
            continue
        seen.add(lc.featureName)

        elements = []

        for f in gws.u.require(lc.model).fields:
            if user.can_read(f):
                elements.append(
                    tag(
                        'XSD:element',
                        {
                            'maxOccurs': '1',
                            'minOccurs': '0',
                            'nillable': 'false' if f.isRequired else 'true',
                            'name': f.name,
                            'type': _xsd_type(f),
                        },
                    )
                )

        type_name = f'{lc.featureName}Type'

        if ns.extendsGml:
            type_def = tag(
                'XSD:complexContent',
                tag('XSD:extension', {'base': 'gml:AbstractFeatureType'}, tag('XSD:sequence', elements)),
            )
        else:
            type_def = tag('XSD:complexContent', tag('XSD:sequence', elements))

        schema.append(tag('XSD:complexType', {'name': type_name}, type_def))

        atts = {
            'name': lc.featureName,
            'type': ns.prefix + ':' + type_name,
        }
        if ns.extendsGml:
            atts['substitutionGroup'] = 'gml:AbstractFeature'

        schema.append(tag('XSD:element', atts))

    return schema, opts


def _xsd_type(f: gws.ModelField) -> str:
    """Return the XSD type for a model field."""

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
