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

    def find_source_features(self, args, source_layers):
        ps = gws.gis.ows.client.prepare_wms_search(
            args,
            source_layers,
            version=self.version,
            force_crs=self.forceCrs,
            always_xy=self.alwaysXY,
        )
        if not ps:
            return []

        op = self.get_operation(gws.OwsVerb.GetFeatureInfo)
        if not op:
            return []

        if op.preferredFormat:
            ps.params.setdefault('INFO_FORMAT', op.preferredFormat)

        text = gws.gis.ows.request.get_text(
            self.request_args_for_operation(op, params=ps.params))

        print(text)

        features = gws.gis.ows.featureinfo.parse(text, default_crs=ps.bounds.crs, always_xy=self.alwaysXY)

        if features is None:
            gws.log.debug(f'WMS NOT_PARSED params={ps.params!r}')
            return []
        gws.log.debug(f'WMS FOUND={len(features)} params={ps.params!r}')

        for f in features:
            if f.shape:
                f.shape = f.shape.transformed_to(args.shapes[0].crs)

        return features
