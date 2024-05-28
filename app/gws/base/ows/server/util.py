"""Server utilities."""

from typing import Optional, cast

import gws
import gws.base.layer.core
import gws.base.model
import gws.base.web
import gws.gis.extent
import gws.gis.render
import gws.lib.date
import gws.lib.mime
import gws.lib.image
import gws.lib.uom
import gws.lib.xmlx as xmlx

from . import core


# @TODO caps trees should be cached for public layers


def layer_caps_tree(sr: core.ServiceRequest, root_layer: Optional[gws.Layer] = None) -> core.LayerCapsTree:
    lct = core.LayerCapsTree(root=None, roots=[], leaves=[])

    if root_layer:
        _collect_layer_caps(sr, lct, root_layer)
    elif sr.project:
        _collect_layer_caps(sr, lct, sr.project.map.rootLayer)
    return lct


def _collect_layer_caps(
        sr: core.ServiceRequest,
        lct: core.LayerCapsTree,
        layer: gws.Layer,
        parent_lc: Optional[core.LayerCaps] = None,
):
    if not sr.req.user.can_read(layer) or not layer.isEnabledForOws:
        return

    lc = layer_caps_for_layer(layer, sr.req.user, [b.crs for b in sr.service.supportedBounds])

    if layer.isGroup:
        for la in layer.layers:
            _collect_layer_caps(sr, lct, la, lc)
        if not lc.children:
            return
    else:
        lct.leaves.append(lc)

    if parent_lc:
        lc.ancestors = parent_lc.ancestors + [parent_lc]
        parent_lc.children.append(lc)
    else:
        lct.roots.append(lc)


def layer_caps_for_layer(layer: gws.Layer, user: gws.User, supported_crs: Optional[list[gws.Crs]] = None) -> core.LayerCaps:
    lc = core.LayerCaps(ancestors=[], children=[])

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

    lc.hasLegend = layer.hasLegend or any(c.hasLegend for c in lc.children)
    lc.hasSearch = layer.isSearchable or any(c.hasSearch for c in lc.children)

    scales = [gws.lib.uom.res_to_scale(r) for r in layer.resolutions]
    lc.minScale = int(min(scales))
    lc.maxScale = int(max(scales))

    if supported_crs:
        lc.bounds = [
            gws.Bounds(
                crs=crs,
                extent=gws.gis.extent.transform_from_wgs(layer.wgsExtent, crs)
            )
            for crs in supported_crs
        ]
    else:
        lc.bounds = []

    return lc


def layer_caps_by_layer_name(lct: core.LayerCapsTree, names: Optional[str | list[str]] = None, with_ancestors=False) -> list[core.LayerCaps]:
    return _leaves(lct, layer_name_matches, names, with_ancestors)


def layer_caps_by_feature_name(lct: core.LayerCapsTree, names: Optional[str | list[str]] = None, with_ancestors=False) -> list[core.LayerCaps]:
    return _leaves(lct, feature_name_matches, names, with_ancestors)


def _leaves(lct, fn, names, with_ancestors):
    if not names:
        return []

    seen = set()
    lcs = []

    for name in gws.u.to_list(names):
        if name in seen:
            continue
        seen.add(name)
        for lc in lct.leaves:
            if fn(lc, name):
                lcs.append(lc)
            elif with_ancestors and any(fn(lc_a, name) for lc_a in lc.ancestors):
                lcs.append(lc)

    return lcs


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


##

def empty_feature_collection(sr: core.ServiceRequest, results: list[gws.SearchResult]) -> core.FeatureCollection:
    return core.FeatureCollection(
        members=[],
        timestamp=gws.lib.date.now_iso(with_tz=False),
        numMatched=len(results),
        numReturned=0,
    )


def feature_collection(sr: core.ServiceRequest, lcs: list[core.LayerCaps], results: list[gws.SearchResult]) -> core.FeatureCollection:
    fc = core.FeatureCollection(
        members=[],
        timestamp=gws.lib.date.now_iso(with_tz=False),
        numMatched=len(results),
        # @TODO paging
        numReturned=len(results),
    )

    for r in results:
        r.feature.transform_to(sr.targetCrs)
        fc.members.append(core.FeatureCollectionMember(
            feature=r.feature,
            layer=r.layer,
            layerCaps=_caps_for_layer(lcs, r.layer)
        ))

    return fc


def _caps_for_layer(lcs: list[core.LayerCaps], layer: gws.Layer):
    for lc in lcs:
        if lc.layer == layer:
            return lc


def one_of_params(sr: core.ServiceRequest, *names):
    for name in names:
        s = sr.req.param(name)
        if s:
            return s


def service_url_path(service: gws.OwsService, project: Optional[gws.Project] = None) -> str:
    return gws.u.action_url_path('owsService', serviceUid=service.uid, projectUid=project.uid if project else None)


def render_map_bbox(sr: core.ServiceRequest, lcs: list[core.LayerCaps]) -> gws.ContentResponse:
    # @TODO image formats

    try:
        px_width = int(sr.req.param('width'))
    except:
        px_width = 0
    try:
        px_height = int(sr.req.param('height'))
    except:
        px_height = 0

    if not px_width:
        raise gws.base.web.error.BadRequest(f'invalid WIDTH')
    if not px_height:
        raise gws.base.web.error.BadRequest(f'invalid HEIGHT')

    transparent = False
    s = sr.req.param('transparent', '').lower()
    if s == 'true':
        transparent = True
    elif s == 'false':
        transparent = False
    elif s:
        raise gws.base.web.error.BadRequest(f'invalid TRANSPARENT')

    planes = [
        gws.MapRenderInputPlane(type=gws.MapRenderInputPlaneType.imageLayer, layer=lc.layer)
        for lc in lcs
    ]

    mri = gws.MapRenderInput(
        backgroundColor=None if transparent else 0,
        bbox=sr.bounds.extent,
        crs=sr.bounds.crs,
        mapSize=(px_width, px_height, gws.Uom.px),
        planes=planes,
        project=sr.project,
        user=sr.req.user,
    )

    mro = gws.gis.render.render_map(mri)

    if mro.planes and mro.planes[0].image:
        content = mro.planes[0].image.to_bytes()
    else:
        content = gws.lib.image.PIXEL_PNG8

    return gws.ContentResponse(mime=gws.lib.mime.PNG, content=content)


def xml_schema(lcs: list[core.LayerCaps]) -> gws.XmlElement:
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
        gml = gws.lib.xmlx.namespace.get('gml3')
        tag.append(['xsd:import', {'namespace': gml.uri, 'schemaLocation': gml.schemaLocation}])

    for lc in lcs:
        elements = []
        for f in lc.model.fields:
            # if user.can_read(f):
            if 1:
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
            'name': f'{lc.featureQname}',
            'type': type_name,
        }
        if ns.extendsGml:
            atts['substitutionGroup'] = 'gml:AbstractFeature'

        tag.append(['xsd:complexType', {'name': type_name}, type_def])
        tag.append(['xsd:element', atts])

    return gws.lib.xmlx.tag(*tag)


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
