import time

import gws
import gws.auth.api
import gws.web
import gws.config
import gws.tools.net
import gws.gis.feature
import gws.common.layer
import gws.gis.cache
import gws.tools.misc
import gws.tools.json2

import gws.types as t


class RenderBboxParams(t.Params):
    bbox: t.Extent
    dpi: t.Optional[int]
    height: int
    layers: t.Optional[t.List[str]]
    layerUid: str
    width: int


class RenderXyzParams(t.Params):
    layerUid: str
    x: int
    y: int
    z: int


class RenderLegendParams(t.Params):
    layerUid: str


class DescribeLayerParams(t.Params):
    layerUid: str


class GetFeaturesParams(t.Params):
    bbox: t.Optional[t.Extent]
    layerUid: str
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class Config(t.WithTypeAndAccess):
    """Map rendering action"""
    pass


_GET_FEATURES_LIMIT = 0


class Object(gws.ActionObject):

    def api_render_bbox(self, req: gws.web.AuthRequest, p: RenderBboxParams) -> t.HttpResponse:
        """Render a part of the map inside a bounding box"""

        layer: t.LayerObject = req.require('gws.ext.layer', p.layerUid)

        cp = {}
        if p.layers:
            cp['layers'] = p.layers
        if p.dpi:
            cp['dpi'] = p.dpi

        bbox = [round(n, 2) for n in p.bbox]

        ts = time.time()

        try:
            img = layer.render_bbox(bbox, p.width, p.height, **cp)
        except:
            gws.log.exception()
            img = gws.tools.misc.Pixels.png8

        gws.log.debug('RENDER_PROFILE: %s - %s - %.2f' % (p.layerUid, repr(bbox), time.time() - ts))

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_render_xyz(self, req: gws.web.AuthRequest, p: RenderXyzParams) -> t.HttpResponse:
        """Render an XYZ tile"""

        layer = req.require('gws.ext.layer', p.layerUid)

        ts = time.time()
        img = None

        try:
            img = layer.render_xyz(p.x, p.y, p.z)
        except:
            gws.log.exception()

        gws.log.debug('RENDER_PROFILE: %s - %s %s %s - %.2f' % (p.layerUid, p.x, p.y, p.z, time.time() - ts))

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        if img and layer.is_public and layer.has_cache:
            gws.gis.cache.store_in_web_cache(layer, p.x, p.y, p.z, img)

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img or gws.tools.misc.Pixels.png8
        })

    def api_render_legend(self, req: gws.web.AuthRequest, p: RenderLegendParams) -> t.HttpResponse:
        """Render a legend for a layer"""

        layer = req.require('gws.ext.layer', p.layerUid)

        if not layer.has_legend:
            raise gws.web.error.NotFound()

        try:
            img = layer.render_legend()
        except:
            gws.log.exception()
            img = gws.tools.misc.Pixels.png8

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_describe_layer(self, req: gws.web.AuthRequest, p: DescribeLayerParams) -> t.HttpResponse:
        layer = req.require('gws.ext.layer', p.layerUid)
        return t.HttpResponse({
            'mimeType': 'text/html',
            'content': layer.description
        })

    def api_get_features(self, req: gws.web.AuthRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        layer: t.LayerObject = req.require('gws.ext.layer', p.layerUid)
        bbox = p.get('bbox') or layer.map.extent
        limit = min(_GET_FEATURES_LIMIT, p.get('limit') or _GET_FEATURES_LIMIT)
        features = layer.get_features(bbox, limit)

        return GetFeaturesResponse({
            'features': [f.convert(target_crs=layer.map.crs).props for f in features],
        })

    def http_get_bbox(self, req: gws.web.AuthRequest, p: RenderBboxParams) -> t.HttpResponse:
        return self.api_render_bbox(req, p)

    def http_get_xyz(self, req: gws.web.AuthRequest, p: RenderXyzParams) -> t.HttpResponse:
        return self.api_render_xyz(req, p)

    def http_get_features(self, req: gws.web.AuthRequest, p: GetFeaturesParams) -> t.HttpResponse:
        res = self.api_get_features(req, p)
        return t.HttpResponse({
            'mimeType': 'application/json',
            'content': gws.tools.json2.to_string(res)
        })

    def http_get_legend(self, req: gws.web.AuthRequest, p: RenderLegendParams) -> t.HttpResponse:
        return self.api_render_legend(req, p)
