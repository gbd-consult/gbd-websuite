import gws.gis.source
import gws.gis.util

import gws.types as t


class WfsServiceConfig(t.Config):
    capsCacheMaxAge: t.Duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[t.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use
    params: dict = {}  #: extra query params
    url: t.Url  #: service url
