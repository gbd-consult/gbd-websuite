"""Map related commands."""

import time

import gws
import gws.base.action
import gws.base.model
import gws.base.layer
import gws.base.template
import gws.base.legend
import gws.base.web.error
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
    layers: t.Optional[t.List[str]]


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
    description: str


class GetFeaturesRequest(gws.Request):
    bbox: t.Optional[gws.Extent]
    layerUid: str
    modelUid: t.Optional[str]
    crs: t.Optional[gws.CrsName]
    resolution: t.Optional[float]
    limit: int = 0
    views: t.Optional[t.List[str]]


class GetFeaturesResponse(gws.Response):
    features: t.List[gws.FeatureProps]


_GET_FEATURES_LIMIT = 10000


class Object(gws.base.action.Object):
    _error_pixel = gws.lib.mime.PNG, gws.lib.image.PIXEL_PNG8

    @gws.ext.command.api('mapGetBox')
    def api_get_box(self, req: gws.IWebRequester, p: GetBoxRequest) -> ImageResponse:
        """Get a part of the map inside a bounding box"""
        mime, content = self._get_box(req, p)
        return ImageResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetBox')
    def http_get_box(self, req: gws.IWebRequester, p: GetBoxRequest) -> gws.ContentResponse:
        mime, content = self._get_box(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapGetXYZ')
    def api_get_xyz(self, req: gws.IWebRequester, p: GetXyzRequest) -> ImageResponse:
        """Get an XYZ tile"""
        mime, content = self._get_xyz(req, p)
        return ImageResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetXYZ')
    def http_get_xyz(self, req: gws.IWebRequester, p: GetXyzRequest) -> gws.ContentResponse:
        mime, content = self._get_xyz(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapGetLegend')
    def api_get_legend(self, req: gws.IWebRequester, p: GetLegendRequest) -> ImageResponse:
        """Get a legend for a layer"""
        mime, content = self._get_legend(req, p)
        return ImageResponse(mime=mime, content=content)

    @gws.ext.command.get('mapGetLegend')
    def http_get_legend(self, req: gws.IWebRequester, p: GetLegendRequest) -> gws.ContentResponse:
        mime, content = self._get_legend(req, p)
        return gws.ContentResponse(mime=mime, content=content)

    @gws.ext.command.api('mapDescribeLayer')
    def describe_layer(self, req: gws.IWebRequester, p: DescribeLayerRequest) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)
        desc = gws.base.template.render(
            layer.templates,
            gws.TemplateRenderInput(args=dict(layer=layer, user=req.user)),
            user=req.user,
            subject='layer.description')
        return DescribeLayerResponse(description=desc.content if desc else '')

    @gws.ext.command.api('mapGetFeatures')
    def api_get_features(self, req: gws.IWebRequester, p: GetFeaturesRequest) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""
        fprops = self._get_features(req, p)
        return GetFeaturesResponse(features=fprops)

    @gws.ext.command.get('mapGetFeatures')
    def http_get_features(self, req: gws.IWebRequester, p: GetFeaturesRequest) -> gws.ContentResponse:
        # @TODO the response should be geojson FeatureCollection

        fprops = self._get_features(req, p)
        js = gws.lib.jsonx.to_string({
            'features': fprops
        })

        return gws.ContentResponse(mime=gws.lib.mime.JSON, content=js)

    ##

    def _get_box(self, req: gws.IWebRequester, p: GetBoxRequest):
        layer = req.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(type='box')

        lri.extraRequest = {}
        if p.layers:
            lri.extraRequest['layers'] = p.layers

        lri.view = gws.gis.render.map_view_from_bbox(
            crs=gws.gis.crs.get(p.crs) or layer.bounds.crs,
            bbox=p.bbox,
            size=(p.width, p.height, gws.Uom.px),
            dpi=gws.lib.uom.OGC_SCREEN_PPI,
            rotation=0
        )

        gws.time_start(f'RENDER_BOX layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
            if lro and lro.content:
                return gws.lib.mime.PNG, lro.content
        except:
            gws.log.exception()
        gws.time_end()

        return self._error_pixel

    def _get_xyz(self, req: gws.IWebRequester, p: GetXyzRequest):
        layer = req.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(type='xyz', x=p.x, y=p.y, z=p.z)
        lro = None

        gws.time_start(f'RENDER_XYZ layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
        except:
            gws.log.exception()
        gws.time_end()

        if not lro:
            return self._error_pixel

        content = lro.content

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        # if content and gws.is_public_object(layer) and layer.has_cache:
        #     path = layer.url_path('tile')
        #     path = path.replace('{x}', str(p.x))
        #     path = path.replace('{y}', str(p.y))
        #     path = path.replace('{z}', str(p.z))
        #     gws.gis.cache.store_in_web_cache(path, content)

        if self.root.app.developer_option('map.annotate_xyz'):
            text = f"{p.z} {p.x} {p.y}"
            img = gws.lib.image.from_bytes(content)
            content = img.add_text(text, x=5, y=5).add_box().to_bytes()

        return gws.lib.mime.PNG, content

    def _get_legend(self, req: gws.IWebRequester, p: GetLegendRequest):
        layer = req.require_layer(p.layerUid)
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

    def _get_features(self, req: gws.IWebRequester, p: GetFeaturesRequest) -> t.List[gws.Props]:
        layer = req.require_layer(p.layerUid)

        model = gws.base.model.locate(layer.models, user=req.user, access=gws.Access.read, uid=p.modelUid)
        if not model:
            raise gws.base.web.error.Forbidden()

        bounds = layer.bounds
        if p.bbox:
            bounds = gws.gis.bounds.from_extent(
                p.bbox,
                gws.gis.crs.get(p.crs) or layer.bounds.crs
            )

        search = gws.SearchArgs(bounds=bounds, limit=_GET_FEATURES_LIMIT)
        features = model.find_features(search, req.user)

        templates = []
        for v in p.views or ['label']:
            tpl = gws.base.template.locate(layer, user=req.user, subject=f'feature.{v}')
            if tpl:
                templates.append(tpl)

        ls = []

        for feature in features:
            feature.compute_values(gws.Access.read, req.user)
            feature.transform_to(bounds.crs)
            feature.render_views(templates, user=req.user, layer=layer)
            ls.append(gws.props(feature, req.user, layer))

        return ls
