"""Map related commands."""

import time

import gws
import gws.types as t
import gws.base.api
import gws.lib.cache
import gws.lib.feature
import gws.lib.json2
import gws.lib.misc
import gws.lib.render
import gws.lib.units


def url_for_render_box(layer_uid) -> str:
    return gws.SERVER_ENDPOINT + '/cmd/mapRenderBox/layerUid/' + layer_uid


def url_for_render_tile(layer_uid, xyz=None) -> str:
    pfx = gws.SERVER_ENDPOINT + '/cmd/mapRenderXyz/layerUid/' + layer_uid
    if xyz:
        return pfx + f'/z/{xyz.z}/x/{xyz.x}/y/{xyz.y}/gws.png'
    return pfx + '/z/{z}/x/{x}/y/{y}/gws.png'


def url_for_render_legend(layer_uid) -> str:
    return gws.SERVER_ENDPOINT + '/cmd/mapRenderLegend/layerUid/' + layer_uid + '/gws.png'


def url_for_get_features(layer_uid) -> str:
    return gws.SERVER_ENDPOINT + '/cmd/mapGetFeatures/layerUid/' + layer_uid


class RenderBoxParams(gws.Params):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[gws.Crs]
    dpi: t.Optional[int]
    layers: t.Optional[t.List[str]]


class RenderXyzParams(gws.Params):
    layerUid: str
    x: int
    y: int
    z: int


class RenderLegendParams(gws.Params):
    layerUid: str


class DescribeLayerParams(gws.Params):
    layerUid: str


class DescribeLayerResponse(gws.Params):
    description: str


class GetFeaturesParams(gws.Params):
    bbox: t.Optional[gws.Extent]
    layerUid: str
    crs: t.Optional[gws.Crs]
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(gws.Response):
    features: t.List[gws.lib.feature.Props]


@gws.ext.Object('action.map')
class Object(gws.base.api.Action):

    @gws.ext.command('api.map.renderBox')
    def api_render_box(self, req: gws.IWebRequest, p: RenderBoxParams) -> gws.ContentResponse:
        """Render a part of the map inside a bounding box"""

        layer = req.require_layer(p.layerUid)
        img = None

        extra_params = {}
        if p.layers:
            extra_params['layers'] = p.layers

        rv = gws.lib.render.view_from_bbox(
            crs=p.crs or layer.map.crs,
            bbox=p.bbox,
            out_size=(p.width, p.height),
            out_size_unit='px',
            dpi=p.dpi or gws.lib.units.OGC_SCREEN_PPI,
            rotation=0
        )

        ts = time.time()
        try:
            img = layer.render_box(rv, extra_params)
        except:
            gws.log.exception()
        gws.log.debug('RENDER_PROFILE: %s - %s - %.2f' % (p.layerUid, repr(rv), time.time() - ts))

        return gws.ContentResponse(mime='image/png', content=img or gws.lib.misc.Pixels.png8)

    @gws.ext.command('get.map.renderBox')
    def http_render_box(self, req: gws.IWebRequest, p: RenderBoxParams) -> gws.ContentResponse:
        return self.api_render_box(req, p)

    @gws.ext.command('api.map.renderXYZ')
    def api_render_xyz(self, req: gws.IWebRequest, p: RenderXyzParams) -> gws.ContentResponse:
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
            gws.lib.cache.store_in_web_cache(url_for_render_tile(layer.uid, p), img)

        return gws.ContentResponse(mime='image/png', content=img or gws.lib.misc.Pixels.png8)

    @gws.ext.command('get.map.renderXYZ')
    def http_render_xyz(self, req: gws.IWebRequest, p: RenderXyzParams) -> gws.ContentResponse:
        return self.api_render_xyz(req, p)

    @gws.ext.command('api.map.renderLegend')
    def api_render_legend(self, req: gws.IWebRequest, p: RenderLegendParams) -> gws.ContentResponse:
        """Render a legend for a layer"""

        path = self._legend_path(req, p)
        content = gws.read_file_b(path) if path else gws.lib.misc.Pixels.png8
        return gws.ContentResponse(mime='image/png', content=content)

    @gws.ext.command('get.map.renderLegend')
    def http_render_legend(self, req: gws.IWebRequest, p: RenderLegendParams) -> gws.ContentResponse:
        path = self._legend_path(req, p)
        if path:
            return gws.ContentResponse(mime='image/png', path=path)
        return gws.ContentResponse(mime='image/png', content=gws.lib.misc.Pixels.png8)

    def _legend_path(self, req: gws.IWebRequest, p: RenderLegendParams):
        layer = req.require_layer(p.layerUid)
        if layer.has_legend:
            try:
                return layer.render_legend_to_path()
            except:
                gws.log.exception()

    @gws.ext.command('api.map.describeLayer')
    def describe_layer(self, req: gws.IWebRequest, p: DescribeLayerParams) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)

        return DescribeLayerResponse(description=layer.description)

    @gws.ext.command('api.map.getFeatures')
    def api_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        layer = req.require_layer(p.layerUid)
        bounds = gws.Bounds(
            crs=p.crs or layer.map.crs,
            extent=p.get('bbox') or layer.map.extent
        )

        found = layer.get_features(bounds, p.get('limit'))

        for f in found:
            f.transform_to(bounds.crs)
            f.apply_templates(keys=['label', 'title'])
            f.apply_data_model()

        return GetFeaturesResponse(features=[f.props for f in found])

    @gws.ext.command('get.map.getFeatures')
    def http_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> gws.ContentResponse:
        res = self.api_get_features(req, p)
        return gws.ContentResponse(mime='application/json', content=gws.lib.json2.to_string(res))
