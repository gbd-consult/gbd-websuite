"""
    Mapproxy config is done in two steps

    1. first, configure the source. For box layers, this is a normal WMS source.
    For tiled layers, we use the 'double cache' technique, see

    https://mapproxy.org/docs/nightly/configuration_examples.html#create-wms-from-existing-tile-server
    https://mapproxy.org/docs/1.11.0/configuration_examples.html#reprojecting-tiles

    Basically, the source is wrapped in a no-store BACK cache, which is then given to the front mpx layer

    2. then, configure the layer. Create the FRONT cache, which is store or no-store, depending on the cache setting.
    Also, configure the _NOCACHE variant for the layer, which skips the DST cache
"""
import gws
from . import core


