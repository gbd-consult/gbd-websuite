import gws
import gws.gis.source
import gws.gis.ows
import gws.gis.extent
import gws.gis.util

import gws.types as t

from . import provider


class WfsServiceConfig(t.Config):
    capsCacheMaxAge: t.Duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[t.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    params: dict = {}  #: extra query params
    url: t.Url  #: service url


def configure_wfs_for(obj: gws.Object, **filter_args):
    obj.url = obj.var('url')
    obj.provider = gws.gis.ows.shared_provider(provider.Object, obj, obj.config)
    obj.source_layers = gws.gis.source.filter_layers(
        obj.provider.source_layers,
        obj.var('sourceLayers'),
        **filter_args)
