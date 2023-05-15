import gws
import gws.base.search.runner
import gws.base.shape
import gws.base.web
import gws.gis.bounds
import gws.gis.crs
import gws.lib.image
import gws.lib.mime
from .. import core

gws.ext.new.owsService('wms')

WMS_130 = '1.3.0'
WMS_111 = '1.1.1'
WMS_110 = '1.1.0'


class Config(core.ServiceConfig):
    """WMS Service configuration"""
    pass


class Object(core.Service):
    protocol = gws.OwsProtocol.WMS
    supported_versions = [WMS_130, WMS_111, WMS_110]
    is_raster_ows = True

    search_max_limit = 100

    @property
    def service_link(self):
        if self.project:
            return gws.Data(url=self.service_url_path(self.project), scheme='OGC:WMS', function='search')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='py',
                path=gws.dirname(__file__) + '/templates/getCapabilities.py',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
                access='all:allow',
            ),
            # NB use the wfs template
            gws.Config(
                type='py',
                path=gws.dirname(__file__) + '/../wfs/templates/getFeature.py',
                subject='ows.GetFeatureInfo',
                mimeTypes=['xml', 'gml', 'gml3'],
                access='all:allow',
            ),
        ]

    @property
    def default_metadata(self):
        return gws.Data(
            inspireDegreeOfConformity='notEvaluated',
            inspireMandatoryKeyword='infoMapAccessService',
            inspireResourceType='service',
            inspireSpatialDataServiceType='view',
            isoScope='dataset',
            isoSpatialRepresentationType='vector',
        )

    ##

    def handle_getcapabilities(self, rd: core.Request):
        # OGC 06-042, 7.2.3.5
        update_sequence = rd.req.param('updatesequence')
        if update_sequence and self.update_sequence and update_sequence >= self.update_sequence:
            raise gws.base.web.error.BadRequest('Wrong update sequence')

        tree = self.layer_caps_tree(rd)
        if not tree.roots:
            gws.log.warning(f'service={self.uid!r}: no layer root found')
            raise gws.base.web.error.NotFound()
        if len(tree.roots) > 1:
            gws.log.warning(f'service={self.uid!r}: multiple layer roots found')
            raise gws.base.web.error.NotFound()

        fmt = rd.req.param('format') or gws.lib.mime.get('xml')

        supported_formats = self.enum_template_formats()
        supported_formats['getmap'] = ['image/png']
        supported_formats['getlegendgraphic'] = ['image/png']

        return self.template_response(rd, gws.OwsVerb.GetCapabilities, fmt, context={
            'layer_root_caps': tree.roots[0],
            'supported_formats': supported_formats,
            'version': self.request_version(rd),
        })

    _getmap_mandatory_params = ['version', 'request', 'layers', 'styles', 'bbox', 'width', 'height', 'format']

    def handle_getmap(self, rd: core.Request):
        if self.with_strict_params:
            for p in self._getmap_mandatory_params:
                if not rd.req.has_param(p):
                    raise gws.base.web.error.BadRequest(f'Required parameter missing: {p}')
            if not rd.req.has_param('crs') and not rd.req.has_param('srs'):
                raise gws.base.web.error.BadRequest('Required parameter missing: CRS')

        trans = rd.req.param('transparent', '').lower()
        if trans:
            if self.with_strict_params and trans not in ('true', 'false'):
                raise gws.base.web.error.BadRequest('Invalid parameter: TRANSPARENT')

        bounds = self._request_bounds(rd)
        if not bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'], self.SCOPE_LAYER)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        return self.render_map_bbox_from_layer_caps_list(rd, lcs, bounds)

    def handle_getlegendgraphic(self, rd: core.Request):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'], self.SCOPE_LAYER)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        out = gws.gis.legend.render(gws.Legend(layers=[lc.layer for lc in lcs if lc.has_legend]))
        return gws.ContentResponse(
            mime=gws.lib.mime.PNG,
            content=gws.gis.legend.to_bytes(out) or gws.lib.image.PIXEL_PNG8)

    def handle_getfeatureinfo(self, rd: core.Request):
        bounds = self._request_bounds(rd)
        if not bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['query_layers'], self.SCOPE_LAYER)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        try:
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
            limit = int(rd.req.param('feature_count', '1'))
            x = int(rd.req.param('i') or rd.req.param('x'))
            y = int(rd.req.param('j') or rd.req.param('y'))
        except:
            raise gws.base.web.error.BadRequest('Invalid parameter')

        request_crs = bounds.crs
        bounds = gws.gis.bounds.transformed_to(bounds, rd.project.map.crs)

        bbox = bounds.extent
        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (x * xres)
        y = bbox[3] - (y * yres)

        point = gws.base.shape.from_geometry({
            'type': 'Point',
            'coordinates': [x, y]
        }, bounds.crs)

        # @TODO: should be a parameter
        pixel_tolerance = 10

        args = gws.SearchQuery(
            project=rd.project,
            layers=[lc.layer for lc in lcs],
            limit=min(limit, self.search_max_limit),
            resolution=xres,
            shapes=[point],
            tolerance=(pixel_tolerance, 'px'),
        )

        features = gws.base.search.runner.run(rd.req, args)

        coll = self.feature_collection(
            rd,
            features,
            lcs,
            target_crs=request_crs,
            populate=True,
            invert_axis_if_geographic=self.request_version(rd) >= WMS_130,
            crs_format=gws.CrsFormat.URN,
        )

        fmt = rd.req.param('info_format', default='gml')
        return self.template_response(rd, gws.OwsVerb.GetFeatureInfo, format=fmt, context={'collection': coll})

    ###

    def _request_bounds(self, rd: core.Request):
        return gws.gis.bounds.from_request_bbox(
            rd.req.param('bbox'),
            gws.gis.crs.get(rd.req.param('crs') or rd.req.param('srs')) or rd.project.map.crs,
            invert_axis_if_geographic=self.request_version(rd) >= WMS_130)
