"""WMS service provider."""

from typing import Optional, cast

import gws
import gws.base.ows.client
import gws.config.util
import gws.lib.crs
import gws.lib.extent
import gws.gis.source


from . import caps


class Config(gws.base.ows.client.provider.Config):
    """Connection to a WMS service."""

    bottomFirst: bool = False
    """The service lists layers in its capabilities from bottom to top."""
    maxRequestPixels: int = 4096
    """Max. size of a single map request in pixels. (added in 8.5)"""


class Object(gws.base.ows.client.provider.Object):
    """WMS service provider.

    Reads the capabilities of a WMS service and runs GetMap and GetFeatureInfo
    requests for the layers, finders and models that use the service.
    """

    protocol = gws.OwsProtocol.WMS

    maxRequestPixels: int
    """Max. width and height of a single GetMap request in pixels; the grabber fetches larger images in chunks."""

    def configure(self):
        self.maxRequestPixels = self.cfg('maxRequestPixels')

        cc = caps.parse(self.get_capabilities(), self.cfg('bottomFirst', default=False))

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version

        self.configure_operations(cc.operations)

    def cache_hash(self):
        return gws.u.sha256([
            super().cache_hash(),
            self.cfg('bottomFirst', default=False),
        ])

    def get_map(self, bounds: gws.Bounds, width: int, height: int, source_layers: list[gws.SourceLayer], mime_type: str) -> bytes:
        """Fetch a map image with GetMap.

        The source layers are given top-first and sent bottom-first, as WMS expects.
        The image is requested with a transparent background.

        Args:
            bounds: Bounds of the image; their CRS is the request CRS.
            width: Image width in pixels.
            height: Image height in pixels.
            source_layers: Source layers to render, topmost first.
            mime_type: Image format to request.

        Returns:
            The image content.

        Raises:
            ``gws.ExternalServiceError``: If the service has no GetMap operation
                or the response is not an image.
        """
        v3 = self.version >= '1.3'

        bbox = bounds.extent
        always_xy = self.alwaysXY or not v3
        if bounds.crs.isYX and not always_xy:
            bbox = gws.lib.extent.swap_xy(bbox)

        layer_names = list(reversed([sl.name for sl in source_layers]))

        params = {
            'BBOX': bbox,
            'CRS' if v3 else 'SRS': bounds.crs.to_string(gws.CrsFormat.epsg),
            'WIDTH': width,
            'HEIGHT': height,
            'LAYERS': layer_names,
            'STYLES': [''] * len(layer_names),
            'FORMAT': mime_type,
            'TRANSPARENT': 'TRUE',
            'VERSION': self.version,
        }

        op = self.get_operation(gws.OwsVerb.GetMap)
        if not op:
            raise gws.ExternalServiceError(f'no GetMap operation in {self.url!r}')

        args = self.prepare_operation(op, params=params)
        res = gws.base.ows.client.request.get(args)
        if not res.content_type.startswith('image/'):
            raise gws.ExternalServiceError(f'GetMap failed: content type {res.content_type!r} url={self.url!r}')
        return res.content

    DEFAULT_GET_FEATURE_LIMIT = 100
    """Value of ``FEATURE_COUNT`` in GetFeatureInfo requests, if the search has no limit."""

    def create_leaf_layer_config(self, source_layers):
        """Create the configuration of a ``wmsflat`` layer for the given source layers.

        Used by the ``wms`` tree layer to create its leaf layers.

        Args:
            source_layers: Source layers to render in the layer.

        Returns:
            A layer configuration that uses this provider.
        """
        return dict(
            type='wmsflat',
            _defaultProvider=self,
            _defaultSourceLayers=source_layers,
        )

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

        try:
            records = gws.base.ows.client.featureinfo.parse(text, default_crs=request_crs, always_xy=self.alwaysXY)
        except gws.Error as exc:
            gws.log.error(f'get_features: parse error: {exc!r}')
            return []

        gws.log.debug(f'get_features: FOUND={len(records)} params={params!r}')

        for rec in records:
            if rec.shape:
                rec.shape = rec.shape.transformed_to(shape.crs)

        return records
