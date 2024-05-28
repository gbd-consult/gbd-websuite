import gws
import gws.base.web
import gws.gis.crs
import gws.gis.bounds
import gws.base.ows.server


gws.ext.new.owsService('wcs')


class Config(gws.base.ows.server.service.Config):
    """WCS Service configuration"""
    pass


class Object(gws.base.ows.server.service.Object):
    protocol = gws.OwsProtocol.WCS
    supportedVersions = ['2.0.1', '1.0.0']
    is_raster_ows = True

    @property
    def service_link(self):
        if self.project:
            return gws.Data(url=self.url_path(self.project), scheme='OGC:WCS', function='search')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='py',
                path=gws.u.dirname(__file__) + '/templates/getCapabilities.cx.py',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
                access=gws.c.PUBLIC,
            ),
            gws.Config(
                type='py',
                path=gws.u.dirname(__file__) + '/templates/describeCoverage.cx.py',
                subject='ows.DescribeCoverage',
                mimeTypes=['xml'],
                access=gws.c.PUBLIC,
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

    # @TODO wcs needs more work

    def handle_getcapabilities(self, rd: core.Request):
        lcs = self.layer_caps_list(rd)
        return self.template_response(rd, gws.OwsVerb.GetCapabilities, context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_describecoverage(self, rd: core.Request):
        lcs = self.layer_caps_list_from_request(rd, ['coverageid', 'coverage', 'identifier'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound()
        return self.template_response(rd, gws.OwsVerb.DescribeCoverage, context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_getcoverage(self, rd: core.Request):
        request_crs = rd.project.map.crs
        p = rd.req.param('srsName')
        if p:
            crs = gws.gis.crs.get(p)
            if not crs:
                raise gws.base.web.error.BadRequest('Invalid CRS')
            request_crs = crs

        bounds = gws.gis.bounds.from_request_bbox(rd.req.param('bbox'), request_crs, invert_axis_if_geographic=True)
        if not bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['coverageid', 'coverage', 'identifier'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        return self.render_map_bbox_from_layer_caps_list(rd, lcs, bounds)
