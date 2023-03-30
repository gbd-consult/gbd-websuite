"""Common functions for OWS client objects."""

import gws
import gws.gis.extent
import gws.gis.source
import gws.gis.crs
import gws.gis.zoom
import gws.types as t


def configure_layers(obj: gws.IOwsClient, provider_class, **filter_args):
    if obj.var('_provider'):
        obj.provider = obj.var('_provider')
        obj.source_layers = obj.var('_source_layers')
    else:
        obj.provider = obj.root.create_shared(provider_class, obj.config)
        slf = gws.merge(
            gws.gis.source.LayerFilter(level=1),
            filter_args,
            obj.var('sourceLayers')
        )
        obj.source_layers = gws.gis.source.filter_layers(obj.provider.sourceLayers, slf)

    if not obj.source_layers:
        raise gws.Error(f'no source layers found for {obj.uid!r}')


def configure_resolutions(obj: gws.IOwsClient, parent_resolultions: list[float] = None):
    zoom = gws.gis.zoom.config_from_source_layers(obj.sourceLayers)
    if zoom:
        la = t.cast(gws.ILayer, obj)
        la.resolutions = gws.gis.zoom.resolutions_from_config(zoom, parent_resolultions)
        return True


def configure_search(obj: gws.IOwsClient, search_class):
    slf = gws.gis.source.LayerFilter(isQueryable=True)
    queryable_layers = gws.gis.source.filter_layers(obj.sourceLayers, slf)
    if queryable_layers:
        t.cast(gws.ILayer, obj).search_providers.append(
            obj.create_required(search_class, gws.Config(
                _provider=obj.provider,
                _source_layers=queryable_layers
            )))
        return True









# def prepared_search(**kwargs) -> PreparedOwsSearch:
#     ps = PreparedOwsSearch(kwargs)
#
#     params = {}
#
#     wms_box_size_m = 500
#     wms_box_size_deg = 1
#     wms_box_size_px = 500
#
#     if ps.protocol == gws.OwsProtocol.WMS:
#         s = wms_box_size_m if ps.point.crs.is_projected else wms_box_size_deg
#         bbox = (
#             ps.point.x - (s >> 1),
#             ps.point.y - (s >> 1),
#             ps.point.x + (s >> 1),
#             ps.point.y + (s >> 1),
#         )
#         ps.bounds = gws.Bounds(crs=ps.point.crs, extent=bbox)
#
#     our_crs = ps.bounds.crs
#
#     ps.request_crs = ps.request_crs or gws.gis.crs.best_match(
#         our_crs,
#         gws.gis.source.supported_crs_list(ps.source_layers))
#
#     bbox = gws.gis.extent.transform(ps.bounds.extent, our_crs, ps.request_crs)
#
#     ps.axis = gws.gis.crs.best_axis(ps.request_crs, protocol=ps.protocol, protocol_version=ps.protocol_version, inverted_crs=ps.inverted_crs)
#     if ps.axis == gws.AXIS_YX:
#         bbox = gws.gis.extent.swap_xy(bbox)
#
#     layer_names = [sl.name for sl in ps.source_layers]
#
#     if ps.protocol == gws.OwsProtocol.WMS:
#         v3 = ps.protocol_version >= '1.3'
#         params = {
#             'BBOX': bbox,
#             'CRS' if v3 else 'SRS': ps.request_crs.to_string(ps.request_crs_format),
#             'WIDTH': wms_box_size_px,
#             'HEIGHT': wms_box_size_px,
#             'I' if v3 else 'X': wms_box_size_px >> 1,
#             'J' if v3 else 'Y': wms_box_size_px >> 1,
#             'LAYERS': layer_names,
#             'QUERY_LAYERS': layer_names,
#             'STYLES': [''] * len(layer_names),
#             'VERSION': ps.protocol_version,
#         }
#         if ps.limit:
#             params['FEATURE_COUNT'] = ps.limit
#
#     if ps.protocol == gws.OwsProtocol.WFS:
#         v2 = ps.protocol_version >= '2.0.0'
#         params = {
#             'BBOX': bbox,
#             'SRSNAME': ps.request_crs.to_string(ps.request_crs_format),
#             'TYPENAMES' if v2 else 'TYPENAME': layer_names,
#             'VERSION': ps.protocol_version,
#         }
#         if ps.limit:
#             params['COUNT' if v2 else 'MAXFEATURES'] = ps.limit
#
#     ps.params = params
#     return ps
