"""Layer configuration tools."""

import gws
import gws.lib.metadata
import gws.gis.extent
import gws.gis.zoom

from . import core


def metadata(layer: core.Object):
    p = layer.var('metadata')
    if p:
        layer.metadata = gws.lib.metadata.from_config(p)
        return True


def search(layer: core.Object):
    if not layer.var('search.enabled'):
        return True
    p = layer.var('search.finders')
    if p:
        for cfg in p:
            layer.searchMgr.add_finder(layer.searchMgr.create_child(gws.ext.object.finder, cfg))
        return True


def bounds(layer: core.Object):
    p = layer.var('extent')
    if p:
        layer.bounds = gws.Bounds(
            crs=layer.parentBounds.crs,
            extent=gws.gis.extent.from_list(p))
        return True


def resolutions(layer: core.Object):
    p = layer.var('zoom')
    if p:
        layer.resolutions = gws.gis.zoom.resolutions_from_config(p, layer.parentResolutions)
        if not layer.resolutions:
            raise gws.Error(f'layer {layer.uid!r}: no resolutions, config={p!r}, parent={layer.parentResolutions!r}')
        return True


def legend(layer: core.Object):
    p = layer.var('legend')
    if p:
        if p.enabled:
            layer.legend = layer.create_child(gws.ext.object.legend, p)
        return True


def group(layer: core.Object, layer_configs):
    has_resolutions = resolutions(layer)
    has_bounds = bounds(layer)

    ls = []

    for cfg in layer_configs:
        cfg = gws.merge(
            cfg,
            _parentBounds=layer.bounds,
            _parentResolutions=layer.resolutions,
        )
        ls.append(layer.create_child(gws.ext.object.layer, cfg))

    layer.layers = gws.compact(ls)
    if not layer.layers:
        raise gws.Error(f'group {layer.uid!r} is empty')

    if not has_resolutions:
        res = set()
        for la in layer.layers:
            res.update(la.resolutions)
        layer.resolutions = sorted(res)

    if not has_bounds:
        layer.bounds = gws.gis.bounds.union([la.bounds for la in layer.layers])

    if not legend(layer):
        layers_uids = [la.uid for la in layer.layers if la.legend]
        if layers_uids:
            layer.legend = layer.create_child(
                gws.ext.object.legend,
                gws.Config(type='combined', layerUids=layers_uids))

    layer.canRenderBox = any(la.canRenderBox for la in layer.layers)
    layer.canRenderXyz = any(la.canRenderXyz for la in layer.layers)
    layer.canRenderSvg = any(la.canRenderSvg for la in layer.layers)

    layer.supportsRasterServices = any(la.supportsRasterServices for la in layer.layers)
    layer.supportsVectorServices = any(la.supportsVectorServices for la in layer.layers)
