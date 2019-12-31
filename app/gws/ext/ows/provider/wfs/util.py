import gws
import gws.gis.source
import gws.gis.util
import gws.gis.proj

import gws.types as t

from . import provider


def configure_wfs(target: gws.Object, **filter_args):
    target.url = target.var('url')

    target.provider = gws.gis.util.shared_ows_provider(provider.Object, target, target.config)
    target.invert_axis_crs = target.var('invertAxis')
    target.source_layers = gws.gis.source.filter_layers(
        target.provider.source_layers,
        target.var('sourceLayers'),
        **filter_args)


def find_features(obj, bbox, target_crs, limit) -> t.List[t.IFeature]:
    provider_crs = gws.gis.util.best_crs(target_crs, obj.provider.supported_crs)
    if provider_crs != target_crs:
        bbox = gws.gis.proj.transform_extent(bbox, target_crs, provider_crs)

    axis = gws.gis.util.best_axis(provider_crs, obj.invert_axis_crs, 'WFS', obj.provider.version)

    args = t.SearchArgs({
        'axis': axis,
        'bbox': bbox,
        'count': limit,
        'crs': provider_crs,
        'layers': [sl.name for sl in obj.source_layers],
        'point': '',
    })

    gws.log.debug(f'WFS_QUERY: START')
    gws.p(args, d=2)

    fs = obj.provider.find_features(args)

    if fs is None:
        gws.log.debug('WFS_QUERY: NOT_PARSED')
        fs = []
    else:
        gws.log.debug(f'WFS_QUERY: FOUND {len(fs)}')

    if provider_crs != target_crs:
        fs = [f.transform(target_crs) for f in fs]

    return fs
