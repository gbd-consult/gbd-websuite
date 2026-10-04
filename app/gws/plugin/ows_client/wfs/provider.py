"""WFS service provider."""

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
    """Connection to a WFS service."""
    
    withBboxCrs: Optional[bool]
    """Add the CRS to the bounding box parameter."""


class Object(gws.base.ows.client.provider.Object):
    """WFS service provider.

    Reads the capabilities of a WFS service and runs GetFeature requests for
    the layers, finders and models that use the service.
    """

    protocol = gws.OwsProtocol.WFS
    withBboxCrs: bool
    """Append the CRS to the ``BBOX`` parameter. Defaults to ``True`` for WFS 2."""
    isWfs2: bool
    """The service version is 2 or higher."""

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
    """Max. number of features in GetFeature requests, if the search has no limit."""

    def create_leaf_layer_config(self, source_layers):
        """Create the configuration of a ``wfsflat`` layer for the given feature types.

        Used by the ``wfs`` tree layer to create its child layers.

        Args:
            source_layers: Feature types to show in the layer.

        Returns:
            A layer configuration that uses this provider.
        """
        return dict(
            type='wfsflat',
            _defaultProvider=self,
            _defaultSourceLayers=source_layers,
        )

    def get_features(self, search, source_layers):
        bounds = search.bounds
        search_shape = None

        if search.shape:
            geometry_tolerance = 0.0

            if search.tolerance:
                n, u = search.tolerance
                geometry_tolerance = n * (search.resolution or 1) if u == 'px' else n

            search_shape = search.shape.tolerance_polygon(geometry_tolerance)
            bounds = search_shape.bounds()

        if not bounds:
            gws.log.warning('get_features: no bounds or shape given')
            return []

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
