"""Map related commands."""

import time

import gws
import gws.base.action
import gws.base.model
import gws.base.layer
import gws.base.template
import gws.base.legend
import gws.gis.cache
import gws.gis.crs
import gws.gis.bounds
import gws.base.feature
import gws.lib.image
import gws.lib.jsonx
import gws.lib.mime
import gws.gis.render
import gws.lib.uom
import gws.types as t

gws.ext.new.action('map')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class GetBoxRequest(gws.Request):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[gws.CrsName]
    dpi: t.Optional[int]
    layers: t.Optional[list[str]]


class GetXyzRequest(gws.Request):
    layerUid: str
    x: int
    y: int
    z: int


class GetLegendRequest(gws.Request):
    layerUid: str


class ImageResponse(gws.Response):
    content: bytes
    mime: str


class DescribeLayerRequest(gws.Request):
    layerUid: str


class DescribeLayerResponse(gws.Request):
    content: str


class GetFeaturesRequest(gws.Request):
    bbox: t.Optional[gws.Extent]
    layerUid: str
    modelUid: t.Optional[str]
    crs: t.Optional[gws.CrsName]
    resolution: t.Optional[float]
    limit: int = 0
    views: t.Optional[list[str]]


class GetFeaturesResponse(gws.Response):
    features: list[gws.FeatureProps]


_GET_FEATURES_LIMIT = 10000


class Object(gws.base.action.Object):
    _error_pixel = gws.lib.mime.PNG, gws.lib.image.PIXEL_PNG8

    @gws.ext.command.api('mapGetBox')
    def api_get_box(self, req: gws.WebRequester, p: GetBoxRequest) -> ImageResponse:
        """Get a part of the map inside a bounding box"""
        mime, content = self._get_box(req, p)
        return ImageResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetBox')
    def http_get_box(self, req: gws.WebRequester, p: GetBoxRequest) -> gws.ContentResponse:
        mime, content = self._get_box(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapGetXYZ')
    def api_get_xyz(self, req: gws.WebRequester, p: GetXyzRequest) -> ImageResponse:
        """Get an XYZ tile"""
        mime, content = self._get_xyz(req, p)
        return ImageResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetXYZ')
    def http_get_xyz(self, req: gws.WebRequester, p: GetXyzRequest) -> gws.ContentResponse:
        mime, content = self._get_xyz(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapGetLegend')
    def api_get_legend(self, req: gws.WebRequester, p: GetLegendRequest) -> ImageResponse:
        """Get a legend for a layer"""
        mime, content = self._get_legend(req, p)
        return ImageResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetLegend')
    def http_get_legend(self, req: gws.WebRequester, p: GetLegendRequest) -> gws.ContentResponse:
        mime, content = self._get_legend(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapDescribeLayer')
    def describe_layer(self, req: gws.WebRequester, p: DescribeLayerRequest) -> DescribeLayerResponse:
        project = req.user.require_project(p.projectUid)
        layer = req.user.require_layer(p.layerUid)
        tpl = self.root.app.templateMgr.find_template(layer, project, user=req.user, subject='layer.description')

        if not tpl:
            return DescribeLayerResponse(content='')

        res = tpl.render(gws.TemplateRenderInput(
            args={'layer': layer},
            localeUid=p.localeUid,
            user=req.user))
        return DescribeLayerResponse(content=res.content)

    @gws.ext.command.api('mapGetFeatures')
    def api_get_features(self, req: gws.WebRequester, p: GetFeaturesRequest) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        propses = self._get_features(req, p)
        return GetFeaturesResponse(features=propses)

    @gws.ext.command.get('mapGetFeatures')
    def http_get_features(self, req: gws.WebRequester, p: GetFeaturesRequest) -> gws.ContentResponse:
        # @TODO the response should be geojson FeatureCollection

        propses = self._get_features(req, p)
        js = gws.lib.jsonx.to_string({
            'features': propses
        })

        return gws.ContentResponse(mime=gws.lib.mime.JSON, content=js)

    ##

    def _get_box(self, req: gws.WebRequester, p: GetBoxRequest):
        layer = req.user.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(type=gws.LayerRenderInputType.box, user=req.user, extraParams={})

        if p.layers:
            lri.extraParams['layers'] = p.layers

        lri.view = gws.gis.render.map_view_from_bbox(
            crs=gws.gis.crs.get(p.crs) or layer.mapCrs,
            bbox=p.bbox,
            size=(p.width, p.height, gws.Uom.px),
            dpi=gws.lib.uom.OGC_SCREEN_PPI,
            rotation=0
        )

        gws.debug.time_start(f'RENDER_BOX layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
            if lro and lro.content:
                return gws.lib.mime.PNG, lro.content
        except:
            gws.log.exception()
        gws.debug.time_end()

        return self._error_pixel

    def _get_xyz(self, req: gws.WebRequester, p: GetXyzRequest):
        layer = req.user.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(type=gws.LayerRenderInputType.xyz, user=req.user, x=p.x, y=p.y, z=p.z)
        lro = None

        gws.debug.time_start(f'RENDER_XYZ layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
        except:
            gws.log.exception()
        gws.debug.time_end()

        if not lro:
            return self._error_pixel

        content = lro.content

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        # if content and gws.u.is_public_object(layer) and layer.has_cache:
        #     path = layer.url_path('tile')
        #     path = path.replace('{x}', str(p.x))
        #     path = path.replace('{y}', str(p.y))
        #     path = path.replace('{z}', str(p.z))
        #     gws.gis.cache.store_in_web_cache(path, content)

        return gws.lib.mime.PNG, content

    def _get_legend(self, req: gws.WebRequester, p: GetLegendRequest):
        layer = req.user.require_layer(p.layerUid)
        lro = layer.render_legend()
        content = gws.base.legend.output_to_bytes(lro)
        if content:
            return lro.mime, content
        return self._error_pixel

    def _image_response(self, lro: gws.LayerRenderOutput) -> ImageResponse:
        # @TODO content-dependent mime type
        # @TODO in-image errors
        if lro and lro.content:
            return ImageResponse(mime='image/png', content=lro.content)
        return ImageResponse(mime='image/png', content=gws.lib.image.PIXEL_PNG8)

    def _get_features(self, req: gws.WebRequester, p: GetFeaturesRequest) -> list[gws.FeatureProps]:
        layer = req.user.require_layer(p.layerUid)
        project = layer.find_closest(gws.ext.object.project)

        crs = gws.gis.crs.get(p.crs) or layer.mapCrs

        bounds = layer.bounds
        if p.bbox:
            bounds = gws.gis.bounds.from_extent(p.bbox, crs)

        search = gws.SearchQuery(
            bounds=bounds,
            project=project,
            layers=[layer],
            limit=_GET_FEATURES_LIMIT
        )

        features = layer.get_features_for_view(search, req.user)
        if not features:
            return []

        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.render, user=req.user)
        model = features[0].model
        return [model.feature_to_view_props(f, mc) for f in features]
