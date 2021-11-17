"""WMS provder."""

import gws
import gws.gis.ows
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
        self.version = cc.version
        self.operations.extend(cc.operations)

    def find_features(self, args, source_layers):
        if not args.shapes:
            return []

        shape = args.shapes[0]
        if shape.geometry_type != gws.GeometryType.point:
            return []

        ps = gws.gis.ows.client.prepared_search(
            inverted_crs=self.inverted_crs,
            limit=args.limit,
            point=shape,
            protocol=self.protocol,
            protocol_version=self.version,
            request_crs=self.force_crs,
            source_layers=source_layers,
        )

        params = gws.merge(ps.params, args.params)

        fmt = self.preferred_formats.get(gws.OwsVerb.GetFeatureInfo)
        if fmt:
            params.setdefault('INFO_FORMAT', fmt)

        op_args = self.operation_args(gws.OwsVerb.GetFeatureInfo, params=params)
        text = gws.gis.ows.request.get_text(**op_args)
        features = gws.gis.ows.formats.read(text, crs=ps.request_crs, axis=ps.axis)

        if features is None:
            gws.log.debug(f'WMS NOT_PARSED params={params!r}')
            return []
        gws.log.debug(f'WMS FOUND={len(features)} params={params!r}')

        return [f.transform_to(shape.crs) for f in features]
