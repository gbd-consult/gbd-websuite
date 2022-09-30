"""WMS provder."""

import gws
import gws.lib.metadata
import gws.gis.ows
import gws.gis.crs
import gws.types as t

from . import caps
from .. import core
from .. import featureinfo

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
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version

        self.configure_operations(cc.operations)

    def find_features(self, args, source_layers):
        if not args.shapes:
            return []

        shape = args.shapes[0]
        if shape.type != gws.GeometryType.point:
            return []

        ps = gws.gis.ows.client.prepared_search(
            limit=args.limit,
            point=shape,
            protocol=self.protocol,
            protocol_version=self.version,
            request_crs=self.forceCrs,
            request_crs_format=gws.CrsFormat.EPSG,
            source_layers=source_layers,
        )

        params = gws.merge(ps.params, args.params)

        op = self.operation(gws.OwsVerb.GetFeatureInfo)
        if op.preferredFormat:
            params.setdefault('INFO_FORMAT', op.preferredFormat)

        text = gws.gis.ows.request.get_text(
            self.request_args_for_operation(op, params=params))
        features = featureinfo.parse(text, crs=ps.request_crs, axis=ps.axis)

        if features is None:
            gws.log.debug(f'WMS NOT_PARSED params={params!r}')
            return []
        gws.log.debug(f'WMS FOUND={len(features)} params={params!r}')

        return [f.transform_to(shape.crs) for f in features]
