from typing import Optional, cast

import gws
import gws.base.legend
import gws.base.ows.server as server
import gws.base.shape
import gws.base.web
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.lib.image
import gws.lib.metadata
import gws.lib.mime

gws.ext.new.owsService('wms')

WMS_130 = '1.3.0'
WMS_111 = '1.1.1'
WMS_110 = '1.1.0'

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/templates/getCapabilities.cx.py',
        subject='ows.GetCapabilities',
        mimeTypes=[gws.lib.mime.XML],
        access=gws.c.PUBLIC,
    ),
    # NB use the wfs template with GML2 (qgis doesn't understand GML3 for WMS)
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/../wfs/templates/getFeature2.cx.py',
        subject='ows.GetFeatureInfo',
        mimeTypes=[
            gws.lib.mime.GML2,
            gws.lib.mime.GML,
            gws.lib.mime.XML,
        ],
        access=gws.c.PUBLIC,
    )
]

_DEFAULT_METADATA = gws.Metadata(
    name='WMS',
    inspireDegreeOfConformity='notEvaluated',
    inspireMandatoryKeyword='infoMapAccessService',
    inspireResourceType='service',
    inspireSpatialDataServiceType='view',
    isoServiceFunction='search',
    isoScope='dataset',
    isoSpatialRepresentationType='vector',
)


class Config(server.service.Config):
    """WMS Service configuration"""
    pass


class Object(server.service.Object):
    protocol = gws.OwsProtocol.WMS
    supportedVersions = [WMS_130, WMS_111, WMS_110]

    isRasterService = True

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.lib.metadata.merge(_DEFAULT_METADATA, self.metadata)

    def configure_operations(self):
        super().configure_operations()

        self.supportedOperations.append(gws.OwsOperation(
            verb=gws.OwsVerb.GetMap, formats=[gws.lib.mime.PNG],
        ))
        self.supportedOperations.append(gws.OwsOperation(
            verb=gws.OwsVerb.GetLegendGraphic, formats=[gws.lib.mime.PNG],
        ))

    ##

    def activate(self):
        self.handlers = {
            gws.OwsVerb.GetCapabilities: self.handle_getcapabilities,
            gws.OwsVerb.GetMap: self.handle_getmap,
            gws.OwsVerb.GetLegendGraphic: self.handle_getlegendgraphic,
            gws.OwsVerb.GetFeatureInfo: self.handle_getfeatureinfo,
        }

    ##

    def init_service_request(self, req):
        sr = super().init_service_request(req)

        sr.crs = self.requested_crs(sr, 'crs', 'srs') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = sr.version < WMS_130
        sr.bounds = self.requested_bounds(sr, 'bbox')

        if sr.verb in {gws.OwsVerb.GetMap, gws.OwsVerb.GetFeatureInfo} and not sr.bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        return sr

    def requested_layer_caps(self, sr: server.ServiceRequest, lct: server.LayerCapsTree):
        s = server.util.one_of_params(sr, 'layer', 'layers')
        if not s:
            return lct.leaves

        lcs = server.util.layer_caps_by_layer_name(lct, gws.u.to_list(s), with_ancestors=True)
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')
        return lcs

    ##

    def handle_getcapabilities(self, sr: server.ServiceRequest):
        lct = server.util.layer_caps_tree(sr, self.rootLayer)
        if len(lct.roots) == 0:
            gws.log.warning(f'ows {self.uid=}: no root found')
            raise gws.base.web.error.NotFound()
        lct.root = lct.roots[0]

        return self.template_response(
            sr,
            format=sr.req.param('format') or gws.lib.mime.XML,
            layerCapsTree=lct,
        )

    # @TODO merge with map/action

    def handle_getmap(self, sr: server.ServiceRequest):
        lct = server.util.layer_caps_tree(sr, self.rootLayer)
        lcs = self.requested_layer_caps(sr, lct)

        return server.util.render_map_bbox(sr, lcs)

    def handle_getlegendgraphic(self, sr: server.ServiceRequest):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        lct = server.util.layer_caps_tree(sr, self.rootLayer)
        lcs = self.requested_layer_caps(sr, lct)

        legend = cast(gws.Legend, self.root.create_temporary(
            gws.ext.object.legend,
            type='combined',
            layerUids=[lc.layer.uid for lc in lcs]))

        content = None

        lro = legend.render()
        if lro:
            content = gws.base.legend.output_to_bytes(lro)

        return gws.ContentResponse(
            mime=gws.lib.mime.PNG,
            content=content or gws.lib.image.PIXEL_PNG8
        )

    def handle_getfeatureinfo(self, sr: server.ServiceRequest):
        lct = server.util.layer_caps_tree(sr, self.rootLayer)
        lcs = self.requested_layer_caps(sr, lct)

        try:
            px_width = int(sr.req.param('width'))
            px_height = int(sr.req.param('height'))
            limit = int(sr.req.param('feature_count', '1'))
            x = int(sr.req.param('i') or sr.req.param('x'))
            y = int(sr.req.param('j') or sr.req.param('y'))
        except:
            raise gws.base.web.error.BadRequest('Invalid parameter')

        bbox = sr.bounds.extent
        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (x * xres)
        y = bbox[3] - (y * yres)

        point = gws.base.shape.from_xy(x, y, sr.crs)

        search = gws.SearchQuery(
            project=sr.project,
            layers=[lc.layer for lc in lcs],
            limit=min(limit, self.searchMaxLimit),
            resolution=xres,
            shape=point,
            tolerance=self.searchTolerance,
        )

        results = self.root.app.searchMgr.run_search(search, sr.req.user)

        return self.template_response(
            sr,
            format=sr.req.param('info_format'),
            featureCollection=server.util.feature_collection(sr, lcs, results)
        )
