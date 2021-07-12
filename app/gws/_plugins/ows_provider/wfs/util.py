import gws
import gws.types as t
import gws.lib.source


class WfsServiceConfig(gws.Config):
    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[gws.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceLayers: t.Optional[gws.lib.source.LayerFilter]  #: source layers to use
    params: dict = {}  #: extra query params
    url: gws.Url  #: service url
