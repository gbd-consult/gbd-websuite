import gws
import gws.gis.source
import gws.gis.util

import gws.types as t

from . import provider

"""

NB: layer order
our configuration lists layers top-to-bottom,
this also applies by default to WMS caps (like in qgis)

for servers with bottom-up caps, set capsLayersBottomUp=True 

the order of GetMap is always bottomUp:

> A WMS shall render the requested layers by drawing the leftmost in the list bottommost, 
> the next one over that, and so on.

http://portal.opengeospatial.org/files/?artifact_id=14416
section 7.3.3.3 

"""


class WmsConfig(t.Config):
    capsCacheMaxAge: t.Duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[t.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    capsLayersBottomUp: bool = False  #: layers are listed from bottom to top in the GetCapabilities document
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: t.Url  #: service url


def configure_wms(target: gws.Object, **filter_args):
    target.url = target.var('url')

    target.provider = gws.gis.util.shared_ows_provider(provider.Object, target, target.config)
    target.axis = target.var('axis')
    target.source_layers = gws.gis.source.filter_layers(
        target.provider.source_layers,
        target.var('sourceLayers'),
        **filter_args)
