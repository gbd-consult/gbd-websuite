"""WFS provider.

References:

- wfs 1.0.0: http://portal.opengeospatial.org/files/?artifact_id=7176 Sec 13.7.3
- wfs 1.1.0: http://portal.opengeospatial.org/files/?artifact_id=8339 Sec 14.7.3
- wfs 2.0.0: http://docs.opengeospatial.org/is/09-025r2/09-025r2.html Sec 11.1.3

See also:

- https://docs.geoserver.org/latest/en/user/services/wfs/reference.html

"""

from typing import Optional, cast

import gws
import gws.base.ows.client
import gws.base.shape
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.lib.extent
import gws.gis.source

from . import caps


class Config(gws.base.ows.client.provider.Config):
    """WFS provider configuration."""
    
    withBboxCrs: Optional[bool]
    """Add CRS to bbox request parameters."""


class Object(gws.base.ows.client.provider.Object):
    protocol = gws.OwsProtocol.WFS
    withBboxCrs: bool
    isWfs2: bool

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version
        self.isWfs2 = self.version >= '2'

        self.configure_operations(cc.operations)

        # use bbox with crs for wfs 2 by default
        # see also comments in qgis/qgswfsfeatureiterator.cpp buildURL
        p = self.cfg('withBboxCrs')
        self.withBboxCrs = self.isWfs2 if p is None else p

    DEFAULT_GET_FEATURE_LIMIT = 100

    def get_features(self, search, source_layers):
        """Perform the WFS GetFeature operation.

        We only do spatial searches here.
        If no bounds and no shapes are given, return all features.
        If a shape is given, find features within its bounds first,
        and filter features on our side.
        This is more performant than WFS spatial ops (at least for qgis),
        and also works without spatial ops support on the provider side.
        """

        bounds = search.bounds
        search_shape = None

        if search.shape:
            geometry_tolerance = 0.0

            if search.tolerance:
                n, u = search.tolerance
                geometry_tolerance = n * (search.resolution or 1) if u == 'px' else n

            search_shape = search.shape.tolerance_polygon(geometry_tolerance)
            bounds = search_shape.bounds()

        request_crs = self.forceCrs or gws.lib.crs.WGS84

        bbox = gws.lib.bounds.transform(bounds, request_crs).extent
        if request_crs.isYX and not self.alwaysXY:
            bbox = gws.lib.extent.swap_xy(bbox)
        bbox = ','.join(str(k) for k in bbox)

        srs = request_crs.urn
        if self.withBboxCrs:
            bbox += ',' + srs

        params = {
            'BBOX': bbox,
            'COUNT' if self.isWfs2 else 'MAXFEATURES': search.limit or self.DEFAULT_GET_FEATURE_LIMIT,
            'SRSNAME': srs,
            'TYPENAMES' if self.isWfs2 else 'TYPENAME': [sl.name for sl in source_layers],
            'VERSION': self.version,
        }

        if search.extraParams:
            params = gws.u.merge(params, gws.u.to_upper_dict(search.extraParams))

        op = self.get_operation(gws.OwsVerb.GetFeature)
        if not op:
            return []

        if op.preferredFormat:
            params.setdefault('OUTPUTFORMAT', op.preferredFormat)

        args = self.prepare_operation(op, params=params)
        text = gws.base.ows.client.request.get_text(args)

        try:
            records = gws.base.ows.client.featureinfo.parse(text, default_crs=request_crs, always_xy=self.alwaysXY)
        except gws.Error as exc:
            gws.log.error(f'get_features: parse error: {exc!r}')
            return []

        gws.log.debug(f'get_features: FOUND={len(records)} params={params!r}')

        for rec in records:
            if rec.shape:
                rec.shape = rec.shape.transformed_to(bounds.crs)

        if not search_shape:
            return records

        filtered = [
            rec for rec in records
            if not rec.shape or rec.shape.intersects(search_shape)
        ]

        gws.log.debug(f'get_features: FILTERED={len(filtered)}')
        return filtered
