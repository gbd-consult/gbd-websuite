import time

import gws
import gws.auth.api
import gws.web
import gws.config
import gws.tools.net
import gws.gis.feature
import gws.gis.layer

import gws.types as t


class RenderBboxParams(t.Data):
    bbox: t.Extent
    width: int
    height: int
    layerUid: str
    layers: t.Optional[t.List[str]]
    dpi: t.Optional[int]


class RenderXyzParams(t.Data):
    layerUid: str
    x: int
    y: int
    z: int


class DescribeLayerParams(t.Data):
    layerUid: str


class GetFeaturesParams(t.Data):
    bbox: t.Optional[t.Extent]
    layerUid: str


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class Config(t.WithTypeAndAccess):
    """map rendering action"""
    pass


# # @TODO return empty images for unsupported layers
#
# def _store_in_front_cache(layer_uid, x, y, z, img):
#     # @TODO: merge with ext/layer/tile/url
#     dir = f'_/cmd/mapHttpGetXyz/layer/{layer_uid}/z/{z}/x/{x}/y/{y}'
#     dir = misc.ensure_dir(dir, gws.WEB_CACHE_DIR)
#     with open(dir + '/t.png', 'wb') as fp:
#         fp.write(img)
#
#
# # def _render_xyz(req, layer_uid, x, y, z):
#
#
# def _render_legend(req, layer_uid):
#     layer: t.LayerObject = req.require('gws.ext.gis.layer', layer_uid)
#     img = layer.render_legend()
#
#     return t.HttpResponse({
#         'mimeType': 'image/png',
#         'content': img
#     })


class Object(gws.Object):

    def api_render_bbox(self, req, p: RenderBboxParams) -> t.HttpResponse:
        """Render a part of the map inside a bounding box"""

        layer: t.LayerObject = req.require('gws.ext.gis.layer', p.layerUid)

        cp = {}
        if p.layers:
            cp['layers'] = p.layers
        if p.dpi:
            cp['dpi'] = p.dpi

        bbox = [round(n, 2) for n in p.bbox]

        ts = time.time()
        img = layer.render_bbox(bbox, p.width, p.height, **cp)
        ts = time.time() - ts

        gws.log.debug('RENDER_PROFILE: %s - %s - %.2f' % (p.layerUid, repr(bbox), ts))

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_render_xyz(self, req, p: RenderXyzParams) -> t.HttpResponse:
        """Render an XYZ tile"""

        layer = req.require('gws.ext.gis.layer', p.layerUid)

        ts = time.time()
        img = layer.render_xyz(p.x, p.y, p.z)
        ts = time.time() - ts

        gws.log.debug('RENDER_PROFILE: %s - %s %s %s - %.2f' % (p.layerUid, p.x, p.y, p.z, ts))

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_describe_layer(self, req, p: DescribeLayerParams) -> t.HttpResponse:
        layer = req.require('gws.ext.gis.layer', p.layerUid)
        return t.HttpResponse({
            'mimeType': 'text/html',
            'content': layer.description()
        })

    def api_get_features(self, req, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        layer = req.require('gws.ext.gis.layer', p.layerUid)
        features = layer.get_features(p.get('bbox'))
        return GetFeaturesResponse({
            'features': [f.props for f in features]
        })

    def http_get_bbox(self, req, p) -> t.HttpResponse:
        ps = {k.lower(): v for k, v in req.params.items()}

        try:
            p = RenderBboxParams({
                'bbox': [float(x) for x in gws.as_list(ps.get('bbox', ''))],
                'dpi': ps.get('dpi'),
                'height': int(ps.get('height')),
                'layers': gws.as_list(ps.get('layers')),
                'layerUid': ps.get('layeruid'),
                'width': int(ps.get('width')),
            })
        except ValueError:
            raise gws.web.error.BadRequest()

        return self.api_render_bbox(req, p)

    def http_get_xyz(self, req, p) -> t.HttpResponse:
        ps = {k.lower(): v for k, v in req.params.items()}

        try:
            p = RenderXyzParams({
                'x': int(ps.get('x')),
                'y': int(ps.get('y')),
                'z': int(ps.get('z')),
                'layerUid': ps.get('layeruid')
            })
        except ValueError:
            raise gws.web.error.BadRequest()

        return self.api_render_xyz(req, p)

    def http_get_legend(self, req, p):
        ps = {k.lower(): v for k, v in req.params.items()}
        layer = req.require('gws.ext.gis.layer', ps.get('layeruid'))
        if layer.legend:
            resp = gws.tools.net.http_request(layer.legend)
            return t.HttpResponse({
                'mimeType': 'image/png',
                'content': resp.content
        })



    def http_get_qgis_legend(self, req, p):
        ps = {k.lower(): v for k, v in req.params.items()}


        # see https://docs.qgis.org/2.18/en/docs/user_manual/working_with_ogc/ogc_server_support.html#getlegendgraphics-request

        ps = {
            'MAP': ps['map'],
            'LAYER': ps['layer'],
            'FORMAT': 'image/png',
            'REQUEST': 'GetLegendGraphic',
            'SERVICE': 'WMS',
            'STYLE': '',
            'VERSION': '1.1.1',
            'BOXSPACE': 0,
            'SYMBOLSPACE': 0,
            'LAYERTITLE': 'false',
            # 'RULELABEL': 'false',
        }

        url = 'http://%s:%s' % (
            gws.config.var('server.qgis.host'),
            gws.config.var('server.qgis.port'))

        resp = gws.tools.net.http_request(url, params=ps)
        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': resp.content
        })
