"""Map related commands."""

import time

import gws
import gws.base.action
import gws.base.layer
import gws.base.legend
import gws.gis.cache
import gws.gis.crs
import gws.base.feature
import gws.lib.image
import gws.lib.json2
import gws.lib.mime
import gws.gis.render
import gws.lib.units as units
import gws.types as t


class GetBoxParams(gws.Request):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[gws.CRS]
    dpi: t.Optional[int]
    layers: t.Optional[t.List[str]]


class GetXyzParams(gws.Request):
    layerUid: str
    x: int
    y: int
    z: int


class GetLegendParams(gws.Request):
    layerUid: str


class DescribeLayerParams(gws.Request):
    layerUid: str


class DescribeLayerResponse(gws.Request):
    description: str


class GetFeaturesParams(gws.Request):
    bbox: t.Optional[gws.Extent]
    layerUid: str
    crs: t.Optional[gws.CRS]
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(gws.Response):
    features: t.List[gws.base.feature.Props]


@gws.ext.object.action('map')
class Object(gws.base.action.Object):
    _error_pixel = gws.lib.mime.PNG, gws.lib.image.PIXEL_PNG8

    @gws.ext.command.api('mapGetBox')
    def api_get_box(self, req: gws.IWebRequester, p: GetBoxParams) -> gws.BytesResponse:
        """Get a part of the map inside a bounding box"""
        mime, content = self._get_box(req, p)
        return gws.BytesResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetBox')
    def http_get_box(self, req: gws.IWebRequester, p: GetBoxParams) -> gws.ContentResponse:
        mime, content = self._get_box(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapGetXYZ')
    def api_get_xyz(self, req: gws.IWebRequester, p: GetXyzParams) -> gws.BytesResponse:
        """Get an XYZ tile"""
        mime, content = self._get_xyz(req, p)
        return gws.BytesResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetXYZ')
    def http_get_xyz(self, req: gws.IWebRequester, p: GetXyzParams) -> gws.ContentResponse:
        mime, content = self._get_xyz(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapGetLegend')
    def api_get_legend(self, req: gws.IWebRequester, p: GetLegendParams) -> gws.BytesResponse:
        """Get a legend for a layer"""
        mime, content = self._get_legend(req, p)
        return gws.BytesResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetLegend')
    def http_get_legend(self, req: gws.IWebRequester, p: GetLegendParams) -> gws.ContentResponse:
        mime, content = self._get_legend(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapDescribeLayer')
    def describe_layer(self, req: gws.IWebRequester, p: DescribeLayerParams) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)
        desc = layer.render_description()
        return DescribeLayerResponse(description=desc.content if desc else '')

    @gws.ext.command.api('mapGetFeatures')
    def api_get_features(self, req: gws.IWebRequester, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""
        found = self._get_features(req, p)
        return GetFeaturesResponse(features=[gws.props(f, req.user, context=self) for f in found])

    @gws.ext.command.get('mapGetFeatures')
    def http_get_features(self, req: gws.IWebRequester, p: GetFeaturesParams) -> gws.ContentResponse:
        # @TODO the response should be geojson FeatureCollection

        found = self._get_features(req, p)
        js = gws.lib.json2.to_string({
            'features': [gws.props(f, req.user, context=self) for f in found]
        })

        return gws.ContentResponse(mime=gws.lib.mime.JSON, content=js)

    ##

    def _get_box(self, req: gws.IWebRequester, p: GetBoxParams):
        layer = req.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(type='box')

        lri.extraParams = {}
        if p.layers:
            lri.extraParams['layers'] = p.layers

        lri.view = gws.gis.render.map_view_from_bbox(
            crs=gws.gis.crs.get(p.crs) or layer.crs,
            bbox=p.bbox,
            size=(p.width, p.height, units.PX),
            dpi=units.OGC_SCREEN_PPI,
            rotation=0
        )

        ts = gws.time_start(f'RENDER_BOX layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
            if lro and lro.content:
                return gws.lib.mime.PNG, lro.content
        except:
            gws.log.exception()
        gws.time_end(ts)

        return self._error_pixel

    def _get_xyz(self, req: gws.IWebRequester, p: GetXyzParams):
        layer = req.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(type='xyz', x=p.x, y=p.y, z=p.z)

        ts = gws.time_start(f'RENDER_XYZ layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
            return gws.lib.mime.PNG, lro.content
        except:
            gws.log.exception()
        gws.time_end(ts)

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        # if content and gws.is_public_object(layer) and layer.has_cache:
        #     path = gws.base.layer.layer_url_path(layer.uid, kind='tile')
        #     path = path.replace('{x}', str(p.x))
        #     path = path.replace('{y}', str(p.y))
        #     path = path.replace('{z}', str(p.z))
        #     gws.gis.cache.store_in_web_cache(path, content)

        return self._error_pixel

    def _get_legend(self, req: gws.IWebRequester, p: GetLegendParams):
        layer = req.require_layer(p.layerUid)
        lro = layer.render_legend()
        content = gws.base.legend.output_to_bytes(lro)
        if content:
            return lro.mime, content
        return self._error_pixel

    def _image_response(self, lro: gws.LayerRenderOutput) -> gws.BytesResponse:
        # @TODO content-dependent mime type
        # @TODO in-image errors
        if lro and lro.content:
            return gws.BytesResponse(mime='image/png', content=lro.content)
        return gws.BytesResponse(mime='image/png', content=gws.lib.image.PIXEL_PNG8)

    def _get_features(self, req: gws.IWebRequester, p: GetFeaturesParams) -> t.List[gws.IFeature]:
        layer = req.require_layer(p.layerUid)
        bounds = gws.Bounds(
            crs=gws.gis.crs.get(p.crs) or layer.map.crs,
            extent=p.get('bbox') or layer.map.extent
        )

        found = layer.get_features(bounds, p.get('limit'))

        for f in found:
            f.transform_to(bounds.crs)
            f.apply_templates(subjects=['label', 'title'])
            f.apply_data_model()

        return found
