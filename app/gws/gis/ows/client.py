"""Common functions for OWS client objects."""

import gws
import gws.gis.extent
import gws.gis.source
import gws.gis.util
import gws.gis.zoom
import gws.types as t


def configure_layers(obj: gws.IOwsClient, provider_class, **filter_args):
    if obj.var('_provider'):
        obj.provider = obj.var('_provider')
        obj.source_layers = obj.var('_source_layers')
    else:
        obj.provider = obj.root.create_object(provider_class, obj.config, shared=True)
        slf = gws.merge(
            gws.gis.source.LayerFilter(level=1),
            filter_args,
            obj.var('sourceLayers')
        )
        obj.source_layers = gws.gis.source.filter_layers(obj.provider.source_layers, slf)

    if not obj.source_layers:
        raise gws.Error(f'no source layers found for {obj.uid!r}')


def configure_zoom(obj: gws.IOwsClient):
    zoom = gws.gis.zoom.config_from_source_layers(obj.source_layers)
    if zoom:
        la = t.cast(gws.ILayer, obj)
        la.resolutions = gws.gis.zoom.resolutions_from_config(zoom, la.resolutions)
        return True


def configure_search(obj: gws.IOwsClient, search_class):
    slf = gws.gis.source.LayerFilter(is_queryable=True)
    queryable_layers = gws.gis.source.filter_layers(obj.source_layers, slf)
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


def prepared_search(**kwargs) -> PreparedOwsSearch:
    ps = PreparedOwsSearch(kwargs)

    params = {}

    wms_box_size_m = 500
    wms_box_size_deg = 1
    wms_box_size_px = 500

    if ps.protocol == gws.OwsProtocol.WMS:
        s = wms_box_size_m if ps.point.crs.is_projected else wms_box_size_deg
        bbox = (
            ps.point.x - (s >> 1),
            ps.point.y - (s >> 1),
            ps.point.x + (s >> 1),
            ps.point.y + (s >> 1),
        )
        ps.bounds = gws.Bounds(crs=ps.point.crs, extent=bbox)

    our_crs = ps.bounds.crs

    ps.request_crs = ps.request_crs or gws.gis.util.best_crs(
        our_crs,
        gws.gis.source.supported_crs_list(ps.source_layers))

    bbox = gws.gis.extent.transform(ps.bounds.extent, our_crs, ps.request_crs)

    ps.axis = gws.gis.util.best_axis(ps.request_crs, ps.protocol, ps.protocol_version, ps.inverted_crs)
    if ps.axis == gws.AXIS_YX:
        bbox = gws.gis.extent.swap_xy(bbox)

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
