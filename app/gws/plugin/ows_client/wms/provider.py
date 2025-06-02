"""WMS provider.

References.

    - OGC 01-068r3: WMS 1.1.1
    - OGC 06-042: WMS 1.3.0

see also https://docs.geoserver.org/latest/en/user/services/wms/reference.html

A note on layer order:

Internally we always list source layers topmost layer first,
which corresponds to the layer tree display.

WMS capabilities are assumed to be top-first by default,
for servers with bottom-first caps, set ``bottomFirst=True``,
in which case the capabilities parser will revert all layer lists.

The order of GetMap is always bottom first:

> A WMS shall render the requested layers by drawing the leftmost in the list bottommost,
> the next one over that, and so on. (OGC 06-042, 7.3.3.3)

therefore when invoking GetMap, our layer lists should be reversed.

"""

from typing import Optional, cast

import gws
import gws.base.ows.client
import gws.config.util
import gws.lib.crs
import gws.lib.extent
import gws.gis.source


from . import caps


class Config(gws.base.ows.client.provider.Config):
    """WMS provider configuration."""

    bottomFirst: bool = False
    """True if layers are listed from bottom to top."""


class Object(gws.base.ows.client.provider.Object):
    protocol = gws.OwsProtocol.WMS

    def configure(self):
        cc = caps.parse(self.get_capabilities(), self.cfg('bottomFirst', default=False))

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version

        self.configure_operations(cc.operations)

    DEFAULT_GET_FEATURE_LIMIT = 100

    def get_features(self, search, source_layers):
        v3 = self.version >= '1.3'

        shape = search.shape
        if not shape or shape.type != gws.GeometryType.point:
            return []

        request_crs = self.forceCrs
        if not request_crs:
            request_crs = gws.lib.crs.best_match(
                shape.crs,
                gws.gis.source.combined_crs_list(source_layers))

        box_size_m = 500
        box_size_deg = 1
        box_size_px = 500

        size = None

        if shape.crs.uom == gws.Uom.m:
            size = box_size_px * search.resolution
        if shape.crs.uom == gws.Uom.deg:
            # @TODO use search.resolution here as well
            size = box_size_deg
        if not size:
            gws.log.debug('cannot request crs {crs!r}, unsupported unit')
            return []

        bbox = (
            shape.x - (size / 2),
            shape.y - (size / 2),
            shape.x + (size / 2),
            shape.y + (size / 2),
        )

        bbox = gws.lib.extent.transform(bbox, shape.crs, request_crs)

        always_xy = self.alwaysXY or not v3
        if request_crs.isYX and not always_xy:
            bbox = gws.lib.extent.swap_xy(bbox)

        layer_names = [sl.name for sl in source_layers]

        params = {
            'BBOX': bbox,
            'CRS' if v3 else 'SRS': request_crs.to_string(gws.CrsFormat.epsg),
            'WIDTH': box_size_px,
            'HEIGHT': box_size_px,
            'I' if v3 else 'X': box_size_px >> 1,
            'J' if v3 else 'Y': box_size_px >> 1,
            'LAYERS': layer_names,
            'QUERY_LAYERS': layer_names,
            'STYLES': [''] * len(layer_names),
            'VERSION': self.version,
            'FEATURE_COUNT': search.limit or self.DEFAULT_GET_FEATURE_LIMIT,
        }

        if search.extraParams:
            params = gws.u.merge(params, gws.u.to_upper_dict(search.extraParams))

        op = self.get_operation(gws.OwsVerb.GetFeatureInfo)
        if not op:
            return []

        if op.preferredFormat:
            params.setdefault('INFO_FORMAT', op.preferredFormat)

        args = self.prepare_operation(op, params=params)
        text = gws.base.ows.client.request.get_text(args)

        fdata = gws.base.ows.client.featureinfo.parse(text, default_crs=request_crs, always_xy=always_xy)

        if fdata is None:
            gws.log.debug(f'get_features: NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'get_features: FOUND={len(fdata)} params={params!r}')

        for fd in fdata:
            if fd.shape:
                fd.shape = fd.shape.transformed_to(shape.crs)

        return fdata
