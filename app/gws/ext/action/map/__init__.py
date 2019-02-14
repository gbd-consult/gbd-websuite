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
    dpi: t.Optional[int]
    height: int
    layers: t.Optional[t.List[str]]
    layerUid: str
    width: int


class RenderXyzParams(t.Data):
    layerUid: str
    x: int
    y: int
    z: int


class RenderLegendParams(t.Data):
    layerUid: str


class DescribeLayerParams(t.Data):
    layerUid: str


class GetFeaturesParams(t.Data):
    bbox: t.Optional[t.Extent]
    layerUid: str
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class Config(t.WithTypeAndAccess):
    """Map rendering action"""
    pass


_GET_FEATURES_LIMIT = 255

# https://commons.wikimedia.org/wiki/File:1x1.png
_1x1_PNG = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x03\x00\x00\x00%\xdbV\xca\x00\x00\x00\x03PLTE\x00\x00\x00\xa7z=\xda\x00\x00\x00\x01tRNS\x00@\xe6\xd8f\x00\x00\x00\nIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82'


class Object(gws.Object):

    def api_render_bbox(self, req, p: RenderBboxParams) -> t.HttpResponse:
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
            img = _1x1_PNG

        ts = time.time() - ts

        gws.log.debug('RENDER_PROFILE: %s - %s - %.2f' % (p.layerUid, repr(bbox), ts))

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_render_xyz(self, req, p: RenderXyzParams) -> t.HttpResponse:
        """Render an XYZ tile"""

        layer = req.require('gws.ext.layer', p.layerUid)

        ts = time.time()

        try:
            img = layer.render_xyz(p.x, p.y, p.z)
        except:
            gws.log.exception()
            img = _1x1_PNG

        ts = time.time() - ts

        gws.log.debug('RENDER_PROFILE: %s - %s %s %s - %.2f' % (p.layerUid, p.x, p.y, p.z, ts))

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_render_legend(self, req, p: RenderLegendParams) -> t.HttpResponse:
        """Render a legend for a layer"""

        layer = req.require('gws.ext.layer', p.layerUid)

        if not layer.has_legend:
            raise gws.web.error.NotFound()

        try:
            img = layer.render_legend()
        except:
            gws.log.exception()
            img = _1x1_PNG

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def api_describe_layer(self, req, p: DescribeLayerParams) -> t.HttpResponse:
        layer = req.require('gws.ext.layer', p.layerUid)
        return t.HttpResponse({
            'mimeType': 'text/html',
            'content': layer.description
        })

    def api_get_features(self, req, p: GetFeaturesParams) -> GetFeaturesResponse:
        """Get a list of features in a bounding box"""

        layer: t.LayerObject = req.require('gws.ext.layer', p.layerUid)
        bbox = p.get('bbox') or layer.map.extent
        limit = min(_GET_FEATURES_LIMIT, p.get('limit') or _GET_FEATURES_LIMIT)
        features = layer.get_features(bbox, limit)

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

        p = RenderLegendParams({
            'layerUid': ps.get('layeruid')
        })

        return self.api_render_legend(req, p)
