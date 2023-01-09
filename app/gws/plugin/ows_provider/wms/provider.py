"""WMS provder."""

import gws
import gws.lib.metadata
import gws.gis.ows
import gws.gis.source
import gws.gis.crs
import gws.gis.extent
import gws.types as t

from . import caps
from .. import core

"""
OGC documents:
    - OGC 01-068r3: WMS 1.1.1
    - OGC 06-042: WMS 1.3.0

see also https://docs.geoserver.org/latest/en/user/services/wms/reference.html

NB: layer order
our configuration lists layers top-to-bottom,
this also applies by default to WMS caps (like in qgis)

for servers with bottom-up caps, set capsLayersBottomUp=True

the order of GetMap is always bottomUp:

> A WMS shall render the requested layers by drawing the leftmost in the list bottommost,
> the next one over that, and so on.

OGC 06-042, 7.3.3.3
"""


class Config(core.ProviderConfig):
    capsLayersBottomUp: bool = False
    """layers are listed from bottom to top in the GetCapabilities document"""


class Object(core.Provider):
    protocol = gws.OwsProtocol.WMS

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version

        self.configure_operations(cc.operations)

