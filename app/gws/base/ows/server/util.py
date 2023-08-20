"""Server utilities."""

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

import gws.types as t

from . import core

DEFAULT_OWS_XML_NAMESPACE = gws.XmlNamespace(
    xmlns='gws',
    uri='http://gbd-websuite.de',
    schemaLocation='',
    version=''
)


# @TODO caps trees should be cached for public layers


def layer_caps_tree(rd: core.Request, root_layer: t.Optional[gws.ILayer] = None) -> core.LayerCapsTree:
    lct = core.LayerCapsTree(root=None, roots=[], leaves=[])

    if root_layer:
        _collect_layer_caps(rd, lct, root_layer)
    elif rd.project:
        r = rd.project.map.rootLayer
        if r.layers:
            _collect_layer_caps(rd, lct, r.layers[0])
    return lct


def _collect_layer_caps(
        rd: core.Request,
        lct: core.LayerCapsTree,
        layer: gws.ILayer,
        parent_lc: t.Optional[core.LayerCaps] = None,
):
    if not rd.req.user.can_read(layer) or not layer.isEnabledForOws:
        return

    lc = core.LayerCaps(ancestors=[], children=[])

    if layer.isGroup:
        for la in layer.layers:
            _collect_layer_caps(rd, lct, la, lc)
        if not lc.children:
            return
    else:
        lct.leaves.append(lc)

    if parent_lc:
        lc.ancestors = parent_lc.ancestors + [parent_lc]
        parent_lc.children.append(lc)
    else:
        lct.roots.append(lc)

    lc.layer = layer
    lc.title = layer.title

    opts = layer.owsOptions

    lc.layerQname = lc.layerName = xmlx.namespace.unqualify_name(opts.layerName)
    lc.featureQname = lc.featureName = xmlx.namespace.unqualify_name(opts.featureName)
    lc.geometryQname = lc.geometryName = xmlx.namespace.unqualify_name(opts.geometryName)

    ns = opts.xmlNamespace
    if ns:
        gws.lib.xmlx.namespace.register(ns)
        lc.layerQname = xmlx.namespace.qualify_name(opts.layerName, ns)
        lc.featureQname = xmlx.namespace.qualify_name(opts.featureName, ns)
        lc.geometryQname = xmlx.namespace.qualify_name(opts.geometryName, ns)

    lc.hasLegend = layer.hasLegend or any(c.hasLegend for c in lc.children)
    lc.hasSearch = layer.isSearchable or any(c.hasSearch for c in lc.children)

    scales = [gws.lib.uom.res_to_scale(r) for r in layer.resolutions]
    lc.minScale = int(min(scales))
    lc.maxScale = int(max(scales))

    lc.bounds = [
        gws.Bounds(
            crs=b.crs,
            extent=gws.gis.extent.transform_from_wgs(layer.wgsExtent, b.crs)
        )
        for b in rd.service.supportedBounds
    ]

    lc.model = gws.base.model.locate(layer.models, user=rd.req.user, access=gws.Access.read)


def layer_caps_by_layer_name(lct: core.LayerCapsTree, names: t.Optional[str | list[str]] = None, with_ancestors=False) -> list[core.LayerCaps]:
    return _leaves(lct, layer_name_matches, names, with_ancestors)


def layer_caps_by_feature_name(lct: core.LayerCapsTree, names: t.Optional[str | list[str]] = None, with_ancestors=False) -> list[core.LayerCaps]:
    return _leaves(lct, feature_name_matches, names, with_ancestors)


def _leaves(lct, fn, names, with_ancestors):
    if not names:
        return []

    seen = set()
    lcs = []

    for name in gws.to_list(names):
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

def empty_feature_collection(rd: core.Request, results: list[gws.SearchResult]) -> core.FeatureCollection:
    return core.FeatureCollection(
        members=[],
        timestamp=gws.lib.date.now_iso(with_tz=False),
        numMatched=len(results),
        numReturned=0,
    )


def feature_collection(rd: core.Request, results: list[gws.SearchResult]) -> core.FeatureCollection:
    fc = core.FeatureCollection(
        members=[],
        timestamp=gws.lib.date.now_iso(with_tz=False),
        numMatched=len(results),
        # @TODO paging
        numReturned=len(results),
    )

    for r in results:
        r.feature.transform_to(rd.targetCrs)
        fc.members.append(core.FeatureCollectionMember(
            feature=r.feature,
            options=r.layer.owsOptions if r.layer else _DEFAULT_OWS_OPTIONS
        ))

    return fc


def one_of_params(rd: core.Request, *names):
    for name in names:
        s = rd.req.param(name)
        if s:
            return s


def service_url_path(service: gws.IOwsService, project: t.Optional[gws.IProject] = None) -> str:
    return gws.action_url_path('owsService', serviceUid=service.uid, projectUid=project.uid if project else None)


def render_map_bbox(rd: core.Request, lcs: list[core.LayerCaps]) -> gws.ContentResponse:
    # @TODO image formats

    try:
        px_width = int(rd.req.param('width'))
    except:
        px_width = 0
    try:
        px_height = int(rd.req.param('height'))
    except:
        px_height = 0

    if not px_width:
        raise gws.base.web.error.BadRequest(f'invalid WIDTH')
    if not px_height:
        raise gws.base.web.error.BadRequest(f'invalid HEIGHT')

    transparent = False
    s = rd.req.param('transparent', '').lower()
    if s == 'true':
        transparent = True
    elif s == 'false':
        transparent = False
    elif s:
        raise gws.base.web.error.BadRequest(f'invalid TRANSPARENT')

    mri = gws.MapRenderInput(
        backgroundColor=None if transparent else 0,
        bbox=rd.bounds.extent,
        crs=rd.bounds.crs,
        mapSize=(px_width, px_height, gws.Uom.px),
        planes=[
            gws.MapRenderInputPlane(type=gws.MapRenderInputPlaneType.imageLayer, layer=lc.layer)
            for lc in lcs
        ]
    )

    mro = gws.gis.render.render_map(mri)

    if mro.planes and mro.planes[0].image:
        content = mro.planes[0].image.to_bytes()
    else:
        content = gws.lib.image.PIXEL_PNG8

    return gws.ContentResponse(mime=gws.lib.mime.PNG, content=content)
