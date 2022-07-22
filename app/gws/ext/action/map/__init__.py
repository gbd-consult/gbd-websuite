"""Map related commands."""

import time

import gws
import gws.common.action
import gws.common.layer
import gws.config
import gws.gis.cache
import gws.gis.feature
import gws.gis.render
import gws.gis.renderview
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


class LoggingConfig(t.Config):
    db: str
    tableName: str
    layerUids: t.List[str]


class Config(t.WithTypeAndAccess):
    """Map rendering action"""
    logging: t.Optional[LoggingConfig]


class Object(gws.common.action.Object):

    def configure(self):
        super().configure()

        self.log_table = None
        p = self.var('logging')
        if p:
            self.db = self.root.find('gws.ext.db.provider', self.var('logging.db'))
            self.log_table = self.var('logging.tableName')
            self.logged_layers_uids = self.var('logging.layerUids')

        if self.log_table:
            with self.db.connect() as conn:
                if not conn.user_can('INSERT', self.log_table):
                    raise ValueError(f'no INSERT acccess to {self.log_table!r}')

    # fs_count integer,
    # fs_ids text

    def log_access(self, action, req: t.IRequest, layer, features=None):
        if self.log_table:
            if not self.logged_layers_uids or \
                    layer.uid in self.logged_layers_uids:
                data = {
                    'date_time': gws.tools.date.now_iso(),
                    'action': action,
                    'layer_uid': layer.uid,
                    'layer_name': layer.title,
                    'login': req.user.uid,
                    'user_name': req.user.display_name,
                    'ip': req.env('REMOTE_ADDR', ''),
                    'fs_count': None,
                    'fs_ids': None
                }
                if features:
                    data['fs_count'] = len(features)
                    data['fs_ids'] = ",".join(f.uid for f in features if f.uid)

                with self.db.connect() as conn:
                    conn.insert_one(self.log_table, 'id', data)

    def api_render_box(self, req: t.IRequest, p: RenderBoxParams) -> t.HttpResponse:
        """Render a part of the map inside a bounding box"""

        layer = req.require_layer(p.layerUid)

        self.log_access('render_box', req, layer)

        img = None

        extra_params = {}
        if p.layers:
            extra_params['layers'] = p.layers

        rv = gws.gis.renderview.from_bbox(
            crs=p.crs or layer.map.crs,
            bbox=p.bbox,
            out_size=(p.width, p.height),
            out_size_unit='px',
            dpi=p.dpi or units.OGC_SCREEN_PPI,
            rotation=0
        )

        ts = time.time()
        try:
            img = layer.render_box(rv, extra_params)
        except:
            gws.log.exception()
        gws.log.debug('RENDER_PROFILE: %s - %s - %.2f' % (p.layerUid, repr(rv), time.time() - ts))

        return t.HttpResponse(mime='image/png', content=img or gws.tools.misc.Pixels.png8)

    def api_render_xyz(self, req: t.IRequest, p: RenderXyzParams) -> t.HttpResponse:
        """Render an XYZ tile"""

        layer = req.require_layer(p.layerUid)

        self.log_access('render_xyz', req, layer)

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
        path = self._legend_path(req, p)
        content = gws.read_file_b(path) if path else gws.tools.misc.Pixels.png8
        return t.HttpResponse(mime='image/png', content=content)

    def http_get_legend(self, req: t.IRequest, p: RenderLegendParams) -> t.Response:
        path = self._legend_path(req, p)
        if path:
            return t.FileResponse(mime='image/png', path=path)
        return t.HttpResponse(mime='image/png', content=gws.tools.misc.Pixels.png8)

    def _legend_path(self, req: t.IRequest, p: RenderLegendParams):
        layer = req.require_layer(p.layerUid)
        if layer.has_legend:
            try:
                return layer.render_legend()
            except:
                gws.log.exception()

    def api_describe_layer(self, req: t.IRequest, p: DescribeLayerParams) -> DescribeLayerResponse:
        layer = req.require_layer(p.layerUid)

        self.log_access('describe_layer', req, layer)

        return DescribeLayerResponse(description=layer.description)

    def api_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        layer = req.require_layer(p.layerUid)


        bounds = t.Bounds(
            crs=p.crs or layer.map.crs,
            extent=p.get('bbox') or layer.map.extent
        )

        found = layer.get_features(bounds, p.get('limit'))

        for f in found:
            f.transform_to(bounds.crs)
            f.apply_templates(keys=['label', 'title'])
            f.apply_data_model()

        self.log_access('get_features', req, layer, features=found)
        return GetFeaturesResponse(features=[f.props for f in found])

    def http_get_box(self, req: t.IRequest, p: RenderBoxParams) -> t.HttpResponse:
        return self.api_render_box(req, p)

    def http_get_xyz(self, req: t.IRequest, p: RenderXyzParams) -> t.HttpResponse:
        return self.api_render_xyz(req, p)

    def http_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> t.HttpResponse:
        res = self.api_get_features(req, p)
        return t.HttpResponse(mime='application/json', content=gws.tools.json2.to_string(res))
