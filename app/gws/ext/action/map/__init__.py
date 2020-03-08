"""Map related commands."""

import time

import gws
import gws.common.action
import gws.common.layer
import gws.config
import gws.gis.cache
import gws.gis.feature
import gws.gis.render
import gws.tools.json2
import gws.tools.misc
import gws.tools.net
import gws.tools.units as units
import gws.web.error

import gws.types as t


class RenderBoxParams(t.Params):
    bbox: t.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[t.Crs]
    dpi: t.Optional[int]
    layers: t.Optional[t.List[str]]


class RenderXyzParams(t.Params):
    layerUid: str
    x: int
    y: int
    z: int


class RenderLegendParams(t.Params):
    layerUid: str


class DescribeLayerParams(t.Params):
    layerUid: str


class DescribeLayerResponse(t.Params):
    description: str


class GetFeaturesParams(t.Params):
    bbox: t.Optional[t.Extent]
    layerUid: str
    crs: t.Optional[t.Crs]
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class Config(t.WithTypeAndAccess):
    """Map rendering action"""
    pass


_FEATURE_FULL_FORMAT_THRESHOLD = 500


class Object(gws.common.action.Object):

    def api_render_box(self, req: t.IRequest, p: RenderBoxParams) -> t.HttpResponse:
        """Render a part of the map inside a bounding box"""

        layer = req.require_layer(p.layerUid)
        img = None

        client_params = {}
        if p.layers:
            client_params['layers'] = p.layers

        rv = gws.gis.render.view_from_bbox(
            crs=p.crs or layer.map.crs,
            bbox=p.bbox,
            out_size=(p.width, p.height),
            out_size_unit='px',
            dpi=p.dpi or units.OGC_SCREEN_PPI,
            rotation=0
        )

        ts = time.time()
        try:
            img = layer.render_box(rv, client_params)
        except:
            gws.log.exception()
        gws.log.debug('RENDER_PROFILE: %s - %s - %.2f' % (p.layerUid, repr(rv), time.time() - ts))

        return t.HttpResponse(mime='image/png', content=img or gws.tools.misc.Pixels.png8)

    def api_render_xyz(self, req: t.IRequest, p: RenderXyzParams) -> t.HttpResponse:
        """Render an XYZ tile"""

        layer = req.require_layer(p.layerUid)
        img = None

        ts = time.time()
        try:
            img = layer.render_xyz(p.x, p.y, p.z)
        except:
            gws.log.exception()
        gws.log.debug('RENDER_PROFILE: %s - %s %s %s - %.2f' % (p.layerUid, p.x, p.y, p.z, time.time() - ts))

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        if img and layer.is_public and layer.has_cache:
            gws.gis.cache.store_in_web_cache(layer, p.x, p.y, p.z, img)

        return t.HttpResponse(mime='image/png', content=img or gws.tools.misc.Pixels.png8)

    def api_render_legend(self, req: t.IRequest, p: RenderLegendParams) -> t.HttpResponse:
        """Render a legend for a layer"""

        layer = req.require_layer(p.layerUid)
        img = None

        if layer.has_legend:
            try:
                img = layer.render_legend()
            except:
                gws.log.exception()

        return t.HttpResponse(mime='image/png', content=img or gws.tools.misc.Pixels.png8)

    def api_describe_layer(self, req: t.IRequest, p: DescribeLayerParams) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)

        return DescribeLayerResponse(description=layer.description)

    def api_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        layer = req.require_layer(p.layerUid)
        bounds = t.Bounds(
            crs=p.crs or layer.map.crs,
            extent=p.get('bbox') or layer.map.extent
        )
        found = layer.get_features(bounds, p.get('limit'))

        # skip full conversion for large amounts of features

        if len(found) > _FEATURE_FULL_FORMAT_THRESHOLD:
            features = [f.transform_to(bounds.crs).minimal_props for f in found]
        else:
            features = [f.transform_to(bounds.crs).apply_converter().props for f in found]

        return GetFeaturesResponse(features=features)

    def http_get_box(self, req: t.IRequest, p: RenderBoxParams) -> t.HttpResponse:
        return self.api_render_box(req, p)

    def http_get_xyz(self, req: t.IRequest, p: RenderXyzParams) -> t.HttpResponse:
        return self.api_render_xyz(req, p)

    def http_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> t.HttpResponse:
        res = self.api_get_features(req, p)
        return t.HttpResponse(mime='application/json', content=gws.tools.json2.to_string(res))

    def http_get_legend(self, req: t.IRequest, p: RenderLegendParams) -> t.HttpResponse:
        return self.api_render_legend(req, p)
