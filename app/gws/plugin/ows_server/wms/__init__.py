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
        path=gws.u.dirname(__file__) + '/templates/getCapabilities.py',
        subject='ows.GetCapabilities',
        mimeTypes=['xml'],
        access=gws.c.PUBLIC,
    ),
    # NB use the wfs template
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/../wfs/templates/getFeature.py',
        subject='ows.GetFeatureInfo',
        mimeTypes=['gml3', 'gml', 'xml'],
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
        return gws.config.util.configure_templates(self, extra=_DEFAULT_TEMPLATES)

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

    def handle_getcapabilities(self, rd: server.Request):
        # OGC 06-042, 7.2.3.5
        s = rd.req.param('updatesequence')
        if s and self.updateSequence and s >= self.updateSequence:
            raise gws.base.web.error.BadRequest('Wrong update sequence')

        lct = server.util.layer_caps_tree(rd, self.rootLayer)
        if len(lct.roots) == 0:
            gws.log.warning(f'ows {self.uid=}: no root found')
            raise gws.base.web.error.NotFound()
        lct.root = lct.roots[0]

        return self.template_response(
            rd,
            gws.OwsVerb.GetCapabilities,
            rd.req.param('format') or gws.lib.mime.XML,
            layerCapsTree=lct,
        )

    # @TODO merge with map/action

    def handle_getmap(self, rd: server.Request):
        self.set_crs_and_bounds(rd)

        lct = server.util.layer_caps_tree(rd, self.rootLayer)
        lcs = self.requested_layer_caps(rd, lct)

        return server.util.render_map_bbox(rd, lcs)

    def handle_getlegendgraphic(self, rd: server.Request):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        lct = server.util.layer_caps_tree(rd, self.rootLayer)
        lcs = self.requested_layer_caps(rd, lct)

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

    def handle_getfeatureinfo(self, rd: server.Request):
        lct = server.util.layer_caps_tree(rd, self.rootLayer)
        lcs = self.requested_layer_caps(rd, lct)

        self.set_crs_and_bounds(rd)

        try:
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
            limit = int(rd.req.param('feature_count', '1'))
            x = int(rd.req.param('i') or rd.req.param('x'))
            y = int(rd.req.param('j') or rd.req.param('y'))
        except:
            raise gws.base.web.error.BadRequest('Invalid parameter')

        bbox = rd.bounds.extent
        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (x * xres)
        y = bbox[3] - (y * yres)

        point = gws.base.shape.from_xy(x, y, rd.crs)

        search = gws.SearchQuery(
            project=rd.project,
            layers=[lc.layer for lc in lcs],
            limit=min(limit, self.searchMaxLimit),
            resolution=xres,
            shape=point,
            tolerance=self.searchTolerance,
        )

        results = self.root.app.searchMgr.run_search(search, rd.req.user)

        fc = server.util.feature_collection(rd, results)
        return self.template_response(
            rd,
            gws.OwsVerb.GetFeatureInfo,
            rd.req.param('info_format', default='gml3'),
            featureCollection=fc)

    ###

    def requested_layer_caps(self, rd: server.Request, lct: server.LayerCapsTree):
        s = server.util.one_of_params(rd, 'layer', 'layers')
        if not s:
            return lct.leaves

        lcs = server.util.layer_caps_by_layer_name(lct, gws.u.to_list(s), with_ancestors=True)
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')
        return lcs

    def set_crs_and_bounds(self, rd: server.Request):
        rd.crs = self.requested_crs(rd) or rd.project.map.bounds.crs
        rd.targetCrs = rd.crs
        rd.alwaysXY = rd.version < WMS_130
        rd.bounds = self.requested_bounds(rd)
