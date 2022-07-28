"""Map related commands."""

import time

import gws
import gws.base.api
import gws.base.layer
import gws.gis.cache
import gws.gis.crs
import gws.gis.feature
import gws.lib.image
import gws.lib.json2
import gws.gis.legend
import gws.lib.mime
import gws.gis.render
import gws.lib.units as units
import gws.types as t


class GetBoxParams(gws.Params):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[gws.CrsId]
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
    crs: t.Optional[gws.CrsId]
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(gws.Response):
    features: t.List[gws.gis.feature.Props]


@gws.ext.object.action('map')
class Object(gws.base.api.action.Object):

    @gws.ext.command.api('mapGetBox')
    def api_get_box(self, req: gws.IWebRequest, p: GetBoxParams) -> gws.BytesResponse:
        """Get a part of the map inside a bounding box"""
        r = self._get_box(req, p)
        return gws.BytesResponse(mime=r.mime, content=r.content)

    @gws.ext.command.get('mapGetBox')
    def http_get_box(self, req: gws.IWebRequest, p: GetBoxParams) -> gws.ContentResponse:
        r = self._get_box(req, p)
        return gws.ContentResponse(mime=r.mime, content=r.content)

    @gws.ext.command.api('mapGetXYZ')
    def api_get_xyz(self, req: gws.IWebRequest, p: GetXyzParams) -> gws.BytesResponse:
        """Get an XYZ tile"""
        r = self._get_xyz(req, p)
        return gws.BytesResponse(mime=r.mime, content=r.content)

    @gws.ext.command.get('mapGetXYZ')
    def http_get_xyz(self, req: gws.IWebRequest, p: GetXyzParams) -> gws.ContentResponse:
        r = self._get_xyz(req, p)
        return gws.ContentResponse(mime=r.mime, content=r.content)

    @gws.ext.command.api('mapGetLegend')
    def api_get_legend(self, req: gws.IWebRequest, p: GetLegendParams) -> gws.BytesResponse:
        """Get a legend for a layer"""
        r = self._get_legend(req, p)
        return gws.BytesResponse(mime=r.mime, content=r.content)

    @gws.ext.command.get('mapGetLegend')
    def http_get_legend(self, req: gws.IWebRequest, p: GetLegendParams) -> gws.ContentResponse:
        r = self._get_legend(req, p)
        return gws.ContentResponse(mime=r.mime, content=r.content)

    @gws.ext.command.api('mapDescribeLayer')
    def describe_layer(self, req: gws.IWebRequest, p: DescribeLayerParams) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)
        return DescribeLayerResponse(description=layer.description)

    @gws.ext.command.api('mapGetFeatures')
    def api_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""
        found = self._get_features(req, p)
        return GetFeaturesResponse(features=[gws.props(f, req.user, context=self) for f in found])

    @gws.ext.command.get('mapGetFeatures')
    def http_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> gws.ContentResponse:
        # @TODO the response should be geojson FeatureCollection

        found = self._get_features(req, p)
        js = gws.lib.json2.to_string({
            'features': [gws.props(f, req.user, context=self) for f in found]
        })

        return gws.ContentResponse(mime=gws.lib.mime.JSON, content=js)

    ##

    def _get_box(self, req: gws.IWebRequest, p: GetBoxParams) -> gws.BytesResponse:
        layer = req.require_layer(p.layerUid)
        content = None

        mri = gws.MapRenderInput(

        )

        extra_params = {}
        if p.layers:
            extra_params['layers'] = p.layers

        view = gws.gis.render.map_view_from_bbox(
            crs=gws.gis.crs.get(p.crs) or layer.map.crs,
            bbox=p.bbox,
            size=(p.width, p.height, units.PX),
            dpi=units.OGC_SCREEN_PPI,
            rotation=0
        )

        ts = gws.time_start(f'RENDER_BOX layer={p.layerUid} view={view!r}')
        try:
            content = layer.render_box(view, extra_params)
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
            gws.gis.cache.store_in_web_cache(path, content)

        return self._image_response(content)

    def _get_legend(self, req: gws.IWebRequest, p: GetLegendParams) -> gws.BytesResponse:
        layer = req.require_layer(p.layerUid)
        content = gws.gis.legend.to_bytes(layer.render_legend_with_cache())
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
            crs=gws.gis.crs.get(p.crs) or layer.map.crs,
            extent=p.get('bbox') or layer.map.extent
        )

        found = layer.get_features(bounds, p.get('limit'))

        for f in found:
            f.transform_to(bounds.crs)
            f.apply_templates(subjects=['label', 'title'])
            f.apply_data_model()

        return found
