"""Map related commands."""

from typing import Optional

import gws
import gws.base.action
import gws.lib.bounds
import gws.lib.crs
import gws.lib.extent
import gws.gis.render
import gws.lib.image
import gws.lib.intl
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.uom

LEGEND_IMAGE_FORMAT = gws.ImageFormat(name='png', mimeTypes=['image/png'], options={})


@gws.ext.config.action('map')
class Config(gws.base.action.Config):
    """Serves map images, tiles, legends and features to the client."""

    pass


@gws.ext.props.action('map')
class Props(gws.base.action.Props):
    pass


class GetBoxRequest(gws.Request):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: Optional[gws.CrsName]
    dpi: Optional[int]
    compositeLayerUids: Optional[list[str]]


class GetTileRequest(gws.Request):
    layerUid: str
    x: int
    y: int
    z: int
    compositeLayerUids: Optional[list[str]]


class GetLegendRequest(gws.Request):
    layerUid: str


class ImageResponse(gws.Response):
    content: bytes
    mimeType: str


class DescribeLayerRequest(gws.Request):
    layerUid: str


class DescribeLayerResponse(gws.Request):
    content: str


class GetFeaturesRequest(gws.Request):
    bbox: Optional[gws.Extent]
    layerUid: str
    modelUid: Optional[str]
    crs: Optional[gws.CrsName]
    resolution: Optional[float]
    limit: int = 0
    views: Optional[list[str]]


class GetFeaturesResponse(gws.Response):
    features: list[gws.FeatureProps]


_GET_FEATURES_LIMIT = 10000
_PIXEL_SIZE_LIMIT = 4096


@gws.ext.object.action('map')
class Object(gws.base.action.Object):
    _empty_pixel = gws.lib.mime.PNG, gws.lib.image.empty_pixel()

    @gws.ext.command.api('mapGetBox')
    def api_get_box(self, req: gws.WebRequester, p: GetBoxRequest) -> ImageResponse:
        """Get a part of the map inside a bounding box"""
        mime_type, content = self._get_box(req, p)
        return ImageResponse(mimeType=mime_type, content=content)

    @gws.ext.command.get('mapGetBox')
    def http_get_box(self, req: gws.WebRequester, p: GetBoxRequest) -> gws.ContentResponse:
        mime_type, content = self._get_box(req, p)
        return gws.ContentResponse(mimeType=mime_type, content=content)

    @gws.ext.command.api('mapGetTile')
    def api_get_tile(self, req: gws.WebRequester, p: GetTileRequest) -> ImageResponse:
        """Get a tile"""
        mime_type, content = self._get_tile(req, p)
        return ImageResponse(mimeType=mime_type, content=content)

    @gws.ext.command.get('mapGetTile')
    def http_get_tile(self, req: gws.WebRequester, p: GetTileRequest) -> gws.ContentResponse:
        mime_type, content = self._get_tile(req, p)
        return gws.ContentResponse(mimeType=mime_type, content=content)

    @gws.ext.command.api('mapGetLegend')
    def api_get_legend(self, req: gws.WebRequester, p: GetLegendRequest) -> ImageResponse:
        """Get a legend for a layer"""
        mime_type, content = self._get_legend(req, p)
        return ImageResponse(mimeType=mime_type, content=content)

    @gws.ext.command.get('mapGetLegend')
    def http_get_legend(self, req: gws.WebRequester, p: GetLegendRequest) -> gws.ContentResponse:
        mime_type, content = self._get_legend(req, p)
        return gws.ContentResponse(mimeType=mime_type, content=content)

    @gws.ext.command.api('mapDescribeLayer')
    def describe_layer(self, req: gws.WebRequester, p: DescribeLayerRequest) -> DescribeLayerResponse:
        project = req.user.require_project(p.projectUid)
        layer = req.user.require_layer(p.layerUid)
        tpl = self.root.app.templateMgr.find_template('layer.description', where=[layer, project], user=req.user)

        if not tpl:
            return DescribeLayerResponse(content='')

        res = tpl.render(gws.TemplateRenderInput(args={'layer': layer}, locale=gws.lib.intl.locale(p.localeUid, project.localeUids), user=req.user))

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
        js = gws.lib.jsonx.to_string({'features': propses})

        return gws.ContentResponse(mimeType=gws.lib.mime.JSON, content=js)

    ##

    def _get_box(self, req: gws.WebRequester, p: GetBoxRequest):
        ok = (
            0 < p.width <= _PIXEL_SIZE_LIMIT and
            0 < p.height <= _PIXEL_SIZE_LIMIT and
            p.width * p.height <= _PIXEL_SIZE_LIMIT * _PIXEL_SIZE_LIMIT
        )  # fmt: skip
        if not ok:
            raise gws.BadRequestError(f'invalid image size')

        layer = req.user.require_layer(p.layerUid)
        crs = gws.lib.crs.get(p.crs) or layer.mapCrs
        lri = gws.LayerRenderInput(
            type=gws.LayerRenderInputType.box,
            targetCrs=crs,
            user=req.user,
            extraParams={},
        )

        bbox = p.bbox
        if crs.isYX and req.param('service') == 'WMS' and req.param('version', '').startswith('1.3'):
            bbox = gws.lib.extent.swap_xy(bbox)

        if p.compositeLayerUids:
            lri.extraParams['compositeLayerUids'] = p.compositeLayerUids

        lri.view = gws.gis.render.map_view_from_bbox(
            crs=crs,
            bbox=bbox,
            size=(p.width, p.height, gws.Uom.px),
            dpi=gws.lib.uom.OGC_SCREEN_PPI,
            rotation=0,
        )

        try:
            lro = layer.render(lri)
            if lro and lro.content:
                return gws.lib.mime.PNG, lro.content
        except Exception:
            gws.log.exception()

        return self._empty_pixel

    def _get_tile(self, req: gws.WebRequester, p: GetTileRequest):
        layer = req.user.require_layer(p.layerUid)
        lri = gws.LayerRenderInput(
            type=gws.LayerRenderInputType.tile,
            targetCrs=layer.mapCrs,
            user=req.user,
            x=p.x,
            y=p.y,
            z=p.z,
            extraParams={},
        )
        if p.compositeLayerUids:
            lri.extraParams['compositeLayerUids'] = p.compositeLayerUids
        lro = None

        gws.debug.time_start(f'RENDER_TILE layer={p.layerUid} lri={lri!r}')
        try:
            lro = layer.render(lri)
        except Exception:
            gws.log.exception()
        gws.debug.time_end()

        if not lro:
            return self._empty_pixel

        content = lro.content

        # for public tiled layers, write tiles to the web cache
        # so they will be subsequently served directly by nginx

        # if content and gws.u.is_public_object(layer) and layer.has_cache:
        #     path = layer.url_path_for('tile')
        #     path = path.replace('{x}', str(p.x))
        #     path = path.replace('{y}', str(p.y))
        #     path = path.replace('{z}', str(p.z))
        #     gws.gis.cache.store_in_web_cache(path, content)

        return gws.lib.mime.PNG, content

    def _get_legend(self, req: gws.WebRequester, p: GetLegendRequest):
        layer = req.user.require_layer(p.layerUid)
        lro = layer.render_legend()
        if lro and lro.image:
            mime_type = LEGEND_IMAGE_FORMAT.mimeTypes[0]
            return mime_type, lro.image.to_bytes(mime_type, LEGEND_IMAGE_FORMAT.options)
        return self._empty_pixel

    def _get_features(self, req: gws.WebRequester, p: GetFeaturesRequest) -> list[gws.FeatureProps]:
        layer = req.user.require_layer(p.layerUid)
        project = layer.find_closest(gws.ext.object.project)

        crs = gws.lib.crs.get(p.crs) or layer.mapCrs

        bounds = layer.bounds
        if p.bbox:
            bounds = gws.lib.bounds.from_extent(p.bbox, crs)

        search = gws.SearchQuery(
            bounds=bounds,
            project=project,
            layers=[layer],
            limit=_GET_FEATURES_LIMIT,
        )

        features = layer.find_features(search, req.user)
        if not features:
            return []

        tpl = self.root.app.templateMgr.find_template(f'feature.label', where=[layer, project], user=req.user)
        if tpl:
            for feature in features:
                feature.render_views([tpl], project=project, layer=self, user=req.user)

        mc = gws.ModelContext(
            op=gws.ModelOperation.read,
            target=gws.ModelReadTarget.map,
            user=req.user,
        )

        return [f.model.feature_to_view_props(f, mc) for f in features]
