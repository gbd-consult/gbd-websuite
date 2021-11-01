"""Miscellaneous GIS-related utilities"""

import re

import gws
import gws.lib.gis.zoom
import gws.lib.extent
import gws.lib.metadata
import gws.lib.crs
import gws.lib.gis.source
import gws.types as t


def configure_ows_client_layers(obj: gws.IOwsClient, provider_class, **filter_args):
    if obj.var('_provider'):
        obj.provider = obj.var('_provider')
        obj.source_layers = obj.var('_source_layers')
    else:
        obj.provider = obj.root.create_object(provider_class, obj.config, shared=True)
        slf = gws.merge(
            gws.lib.gis.source.LayerFilter(level=1),
            filter_args,
            obj.var('sourceLayers')
        )
        obj.source_layers = gws.lib.gis.source.filter_layers(obj.provider.source_layers, slf)

    if not obj.source_layers:
        raise gws.Error(f'no source layers found for {obj.uid!r}')


def configure_ows_client_zoom(obj: gws.IOwsClient):
    zoom = gws.lib.gis.zoom.config_from_source_layers(obj.source_layers)
    if zoom:
        la = t.cast(gws.ILayer, obj)
        la.resolutions = gws.lib.gis.zoom.resolutions_from_config(zoom, la.resolutions)
        return True


def configure_ows_client_search(obj: gws.IOwsClient, search_class):
    slf = gws.lib.gis.source.LayerFilter(is_queryable=True)
    queryable_layers = gws.lib.gis.source.filter_layers(obj.source_layers, slf)
    if queryable_layers:
        t.cast(gws.ILayer, obj).search_providers.append(
            obj.require_child(search_class, gws.Config(
                _provider=obj.provider,
                _source_layers=queryable_layers
            )))
        return True


class PreparedOwsSearch(gws.Data):
    axis: gws.Axis
    bounds: gws.Bounds
    inverted_crs: t.List[gws.ICrs]
    limit: int
    params: t.Dict[str, t.Any]
    point: gws.IShape
    protocol: gws.OwsProtocol
    protocol_version: str
    request_crs: gws.ICrs
    source_layers: t.List[gws.SourceLayer]


def prepared_ows_search(**kwargs) -> PreparedOwsSearch:
    ps = PreparedOwsSearch(kwargs)

    params = {}

    wms_box_size_m = 500
    wms_box_size_px = 500

    if ps.protocol == gws.OwsProtocol.WMS:
        bbox = (
            ps.point.x - (wms_box_size_m >> 1),
            ps.point.y - (wms_box_size_m >> 1),
            ps.point.x + (wms_box_size_m >> 1),
            ps.point.y + (wms_box_size_m >> 1),
        )
        ps.bounds = gws.Bounds(crs=ps.point.crs, extent=bbox)

    our_crs = ps.bounds.crs

    ps.request_crs = ps.request_crs or best_crs(
        our_crs,
        gws.lib.gis.source.supported_crs_list(ps.source_layers))

    bbox = gws.lib.extent.transform(ps.bounds.extent, our_crs, ps.request_crs)

    ps.axis = best_axis(ps.request_crs, ps.protocol, ps.protocol_version, ps.inverted_crs)
    if ps.axis == gws.AXIS_YX:
        bbox = gws.lib.extent.swap_xy(bbox)

    layer_names = [sl.name for sl in ps.source_layers]

    if ps.protocol == gws.OwsProtocol.WMS:
        v3 = ps.protocol_version >= '1.3'
        params = {
            'BBOX': bbox,
            'CRS' if v3 else 'SRS': ps.request_crs.to_string(),
            'WIDTH': wms_box_size_px,
            'HEIGHT': wms_box_size_px,
            'I' if v3 else 'X': wms_box_size_px >> 1,
            'J' if v3 else 'Y': wms_box_size_px >> 1,
            'LAYERS': layer_names,
            'QUERY_LAYERS': layer_names,
            'STYLES': [''] * len(layer_names),
            'VERSION': ps.protocol_version,
        }
        if ps.limit:
            params['FEATURE_COUNT'] = ps.limit

    if ps.protocol == gws.OwsProtocol.WFS:
        v2 = ps.protocol_version >= '2.0.0'
        params = {
            'BBOX': bbox,
            'SRSNAME': ps.request_crs.to_string(),
            'TYPENAMES' if v2 else 'TYPENAME': layer_names,
            'VERSION': ps.protocol_version,
        }
        if ps.limit:
            params['COUNT' if v2 else 'MAXFEATURES'] = ps.limit

    ps.params = params
    return ps


def best_axis(
        crs: gws.ICrs,
        protocol: gws.OwsProtocol,
        protocol_version: str,
        inverted_crs: t.Optional[t.List[gws.ICrs]] = None
) -> gws.Axis:
    # inverted_axis_crs_list - list of projection refs which are known
    # to have an inverted axis for this service
    # crs_ref - projection we're going to use with the service

    if inverted_crs and crs in inverted_crs:
        return gws.AXIS_YX

    # @TODO some logic to guess the axis, based on crs, service protocol and version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
    return gws.AXIS_XY


def best_bounds(
        crs: gws.ICrs,
        supported_bounds: t.List[gws.Bounds],
        prefer_projected=True
) -> gws.Bounds:
    for b in supported_bounds:
        if b.crs == crs:
            return b

    return t.cast(gws.Bounds, _best_crs_or_bounds(crs, supported_bounds, is_bounds=True, prefer_projected=prefer_projected))


def best_crs(
        crs: gws.ICrs,
        supported_crs: t.List[gws.ICrs],
        prefer_projected=True
) -> gws.ICrs:
    if crs in supported_crs:
        return crs

    return t.cast(gws.ICrs, _best_crs_or_bounds(crs, supported_crs, is_bounds=False, prefer_projected=prefer_projected))


def _best_crs_or_bounds(want_crs, supported, is_bounds: bool, prefer_projected: bool):
    # @TODO find a projection with less errors

    # a crs with the same srid?

    for s in supported:
        crs = s.crs if is_bounds else s
        if crs.same_as(want_crs):
            return s

    # webmercator supported?

    for s in supported:
        crs = s.crs if is_bounds else s
        if prefer_projected and crs.srid == gws.lib.crs.c3857:
            gws.log.debug(f'best_crs: using {crs.srid!r} for {want_crs.srid!r}')
            return s

    # first projected crs?

    for s in supported:
        crs = s.crs if is_bounds else s
        if prefer_projected == crs.is_projected:
            gws.log.debug(f'best_crs: using {crs.srid!r} for {want_crs.srid!r}')
            return s

    # return the first one

    for s in supported:
        crs = s.crs if is_bounds else s
        gws.log.debug(f'best_crs: using {crs.srid!r} for {want_crs.srid!r}')
        return s
