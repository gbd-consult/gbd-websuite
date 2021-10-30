"""Map related commands."""

import time

import gws
import gws.base.api
import gws.base.layer
import gws.lib.cache
import gws.lib.feature
import gws.lib.image
import gws.lib.json2
import gws.lib.legend
import gws.lib.mime
import gws.lib.render
import gws.lib.units
import gws.types as t


class GetBoxParams(gws.Params):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[gws.Crs]
    dpi: t.Optional[int]
    layers: t.Optional[t.List[str]]


class GetXyzParams(gws.Params):
    layerUid: str
    x: int
    y: int
    z: int


class GetLegendParams(gws.Params):
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
class Object(gws.base.api.action.Object):

    @gws.ext.command('api.map.getBox')
    def api_get_box(self, req: gws.IWebRequest, p: GetBoxParams) -> gws.BytesResponse:
        """Get a part of the map inside a bounding box"""
        r = self._get_box(req, p)
        return gws.BytesResponse(mime=r.mime, content=r.content)

    @gws.ext.command('get.map.getBox')
    def http_get_box(self, req: gws.IWebRequest, p: GetBoxParams) -> gws.ContentResponse:
        r = self._get_box(req, p)
        return gws.ContentResponse(mime=r.mime, content=r.content)

    @gws.ext.command('api.map.getXYZ')
    def api_get_xyz(self, req: gws.IWebRequest, p: GetXyzParams) -> gws.BytesResponse:
        """Get an XYZ tile"""
        r = self._get_xyz(req, p)
        return gws.BytesResponse(mime=r.mime, content=r.content)

    @gws.ext.command('get.map.getXYZ')
    def http_get_xyz(self, req: gws.IWebRequest, p: GetXyzParams) -> gws.ContentResponse:
        r = self._get_xyz(req, p)
        return gws.ContentResponse(mime=r.mime, content=r.content)

    @gws.ext.command('api.map.getLegend')
    def api_get_legend(self, req: gws.IWebRequest, p: GetLegendParams) -> gws.BytesResponse:
        """Get a legend for a layer"""
        r = self._get_legend(req, p)
        return gws.BytesResponse(mime=r.mime, content=r.content)

    @gws.ext.command('get.map.getLegend')
    def http_get_legend(self, req: gws.IWebRequest, p: GetLegendParams) -> gws.ContentResponse:
        r = self._get_legend(req, p)
        return gws.ContentResponse(mime=r.mime, content=r.content)

    @gws.ext.command('api.map.describeLayer')
    def describe_layer(self, req: gws.IWebRequest, p: DescribeLayerParams) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)
        return DescribeLayerResponse(description=layer.description)

    @gws.ext.command('api.map.getFeatures')
    def api_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""
        found = self._get_features(req, p)
        return GetFeaturesResponse(features=[gws.props(f, req.user, context=self) for f in found])

    @gws.ext.command('get.map.getFeatures')
    def http_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> gws.ContentResponse:
        # @TODO the response should be geojson FeatureCollection
        found = self._get_features(req, p)
        ts = gws.time_start('get_features')
        res = gws.ContentResponse(
            mime=gws.lib.mime.JSON,
            content=gws.lib.json2.to_string({
                'features': [gws.props(f, req.user, context=self) for f in found]
            }))
        gws.time_end(ts)
        return res

    ##

    def _get_box(self, req: gws.IWebRequest, p: GetBoxParams) -> gws.BytesResponse:
        layer = req.require_layer(p.layerUid)
        content = None

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

        ts = gws.time_start(f'RENDER_BOX layer={p.layerUid} view={rv!r}')
        try:
            content = layer.render_box(rv, extra_params)
        except:
            gws.log.exception()
        gws.time_end(ts)

        return self._image_response(content)

    def _get_xyz(self, req: gws.IWebRequest, p: GetXyzParams) -> gws.BytesResponse:
        layer = req.require_layer(p.layerUid)
        content = None

        ts = gws.time_start(f'RENDER_XYZ layer={p.layerUid} xyz={p.x}/{p.y}/{p.z}')
        try:
            content = layer.render_xyz(p.x, p.y, p.z)
        except:
            gws.log.exception()
        gws.time_end(ts)

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        if content and gws.is_public_object(layer) and layer.has_cache:
            path = gws.base.layer.layer_url_path(layer.uid, kind='tile')
            path = path.replace('{x}', str(p.x))
            path = path.replace('{y}', str(p.y))
            path = path.replace('{z}', str(p.z))
            gws.lib.cache.store_in_web_cache(path, content)

        return self._image_response(content)

    def _get_legend(self, req: gws.IWebRequest, p: GetLegendParams) -> gws.BytesResponse:
        layer = req.require_layer(p.layerUid)
        content = gws.lib.legend.to_bytes(layer.render_legend_with_cache())
        return self._image_response(content)

    def _image_response(self, content) -> gws.BytesResponse:
        # @TODO content-dependent mime type
        # @TODO in-image errors
        if content:
            return gws.BytesResponse(mime='image/png', content=content)
        return gws.BytesResponse(mime='image/png', content=gws.lib.image.PIXEL_PNG8)

    def _get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> t.List[gws.IFeature]:
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

        return found
