import gws
import gws.types as t
import gws.lib.metadata
import gws.base.ows.service as ows
import gws.base.search.runner
import gws.lib.bounds
import gws.lib.extent
import gws.lib.gml
import gws.lib.legend
import gws.lib.proj
import gws.lib.render
import gws.lib.shape
import gws.lib.mime
import gws.lib.misc
import gws.lib.os2
import gws.lib.xml2
import gws.base.web.error


class Config(gws.base.ows.service.Config):
    """WCS Service configuration"""
    pass


class Object(ows.Base):

    @property
    def service_link(self):
        if self.project:
            return gws.MetaLink(url=self.url_for_project(self.project), scheme='OGC:WCS', function='search')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wcs/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wcs/templates/describeCoverage.cx',
                subject='ows.DescribeCoverage',
                mimeTypes=['xml'],
            ),
        ]

    @property
    def default_metadata(self):
        return gws.Data(
            inspireDegreeOfConformity=gws.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=gws.MetaInspireMandatoryKeyword.infoMapAccessService,
            inspireResourceType=gws.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=gws.MetaInspireSpatialDataServiceType.view,
            isoScope=gws.MetaIsoScope.dataset,
            isoSpatialRepresentationType=gws.MetaIsoSpatialRepresentationType.vector,
        )

    @property
    def default_name(self):
        return 'WCS'

    ##

    def configure(self):
        

        self.type = 'wcs'
        self.supported_versions = ['2.0.1', '1.0.0']

    ##

    def handle_getcapabilities(self, rd: ows.Request):
        lcs = self.layer_caps_list(rd)
        return self.template_response(rd, 'GetCapabilities', context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_describecoverage(self, rd: ows.Request):
        lcs = self.layer_caps_list_from_request(rd, ['coverageid', 'coverage', 'identifier'])
        if not lcs:
            raise gws.base.web.error.NotFound()
        return self.template_response(rd, 'DescribeCoverage', context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_getcoverage(self, rd: ows.Request):
        bounds = gws.lib.bounds.from_request_bbox(
            rd.req.param('bbox'),
            rd.req.param('crs') or rd.req.param('srs') or rd.project.map.crs,
            invert_axis_if_geographic=True
        )
        if not bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['coverageid', 'coverage', 'identifier'])
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        return self.render_map_bbox_from_layer_caps_list(lcs, bounds, rd)
