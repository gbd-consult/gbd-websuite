import gws
import gws.types as t
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
from .. import core


@gws.ext.Config('ows.service.wcs')
class Config(core.ServiceConfig):
    """WCS Service configuration"""
    pass


@gws.ext.Object('ows.service.wcs')
class Object(core.Service):
    protocol = gws.OwsProtocol.WCS
    supported_versions = ['2.0.1', '1.0.0']

    @property
    def service_link(self):
        if self.project:
            return gws.Data(url=self.service_url_path(self.project), scheme='OGC:WCS', function='search')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/describeCoverage.cx',
                subject='ows.DescribeCoverage',
                mimeTypes=['xml'],
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

    def configure(self):
        pass

    ##

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
        bounds = gws.lib.bounds.from_request_bbox(
            rd.req.param('bbox'),
            rd.req.param('crs') or rd.req.param('srs') or rd.project.map.crs,
            invert_axis_if_geographic=True
        )
        if not bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['coverageid', 'coverage', 'identifier'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        return self.render_map_bbox_from_layer_caps_list(lcs, bounds, rd)
