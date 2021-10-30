"""WMS provder."""

import gws
import gws.lib.metadata
import gws.lib.feature
import gws.lib.gis
import gws.lib.net
import gws.lib.ows
import gws.lib.xml2
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
    capsLayersBottomUp: bool = False  #: layers are listed from bottom to top in the GetCapabilities document


class Object(core.Provider):
    protocol = gws.OwsProtocol.WMS

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs
        self.version = cc.version
        self.operations.extend(cc.operations)

    def find_features(self, args: gws.SearchArgs) -> t.List[gws.IFeature]:
        if not args.shapes:
            return []

        our_crs = args.shapes[0].crs

        ps = gws.lib.gis.prepare_wms_search(
            args.shapes[0],
            protocol_version=self.version,
            force_crs=self.source_crs,
            supported_crs=self.supported_crs,
            invert_axis_crs=self.invert_axis_crs
        )

        if not ps:
            return []

        params = gws.merge(ps.params, {
            'LAYERS': args.source_layer_names,
            'QUERY_LAYERS': args.source_layer_names,
            'STYLES': [''] * len(args.source_layer_names),
        })

        if args.limit:
            params['FEATURE_COUNT'] = args.limit

        params = gws.merge(params, args.params)

        fmt = self.preferred_formats.get(str(gws.OwsVerb.GetFeatureInfo))
        if fmt:
            params.setdefault('INFO_FORMAT', fmt)

        text = gws.lib.ows.request.get_text(**self.operation_args(gws.OwsVerb.GetFeatureInfo, params=params))
        features = gws.lib.ows.formats.read(text, crs=ps.request_crs, axis=ps.axis)

        if features is None:
            gws.log.debug(f'WMS NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'WMS FOUND={len(features)} params={params!r}')
        return [f.transform_to(our_crs) for f in features]
