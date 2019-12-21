import gws.gis.source
import gws.types as t


class SourceLayer(t.SourceLayer):
    # this actually the  wfs 'FeatureType'
    pass


class WfsServiceConfig(t.Config):
    capsCacheMaxAge: t.Duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[t.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: t.Url  #: service url
