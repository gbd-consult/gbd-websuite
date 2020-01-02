import gws
import gws.gis.source
import gws.gis.ows
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


def configure_wms_for(obj: t.IObject, **filter_args):
    obj.url = obj.var('url')
    obj.provider = gws.gis.ows.shared_provider(provider.Object, obj, obj.config)
    obj.source_layers = gws.gis.source.filter_layers(
        obj.provider.source_layers,
        obj.var('sourceLayers'),
        **filter_args)
