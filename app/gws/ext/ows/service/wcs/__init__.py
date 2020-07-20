import io

import gws
import gws.common.metadata
import gws.common.search.runner
import gws.gis.extent
import gws.gis.gml
import gws.gis.legend
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.tools.misc
import gws.tools.os2
import gws.tools.xml2
import gws.tools.mime
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows


class Config(gws.common.ows.service.Config):
    """WCS Service configuration"""
    pass


class Object(ows.Base):

    @property
    def service_link(self):
        return t.MetaLink(url=self.url, scheme='OGC:WCS', function='search')

    @property
    def default_templates(self):
        return [
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wcs/templates/getCapabilities.cx',
                owsRequest='GetCapabilities',
                owsFormat=gws.tools.mime.get('xml'),
            ),
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wcs/templates/describeCoverage.cx',
                owsRequest='DescribeCoverage',
                owsFormat=gws.tools.mime.get('xml'),
            ),
        ]

    @property
    def default_metadata(self):
        return t.Data(
            inspireDegreeOfConformity=t.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=t.MetaInspireKeyword.infoMapAccessService,
            inspireResourceType=t.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=t.MetaInspireSpatialDataServiceType.view,
            isoScope=t.MetaIsoScope.dataset,
            isoSpatialRepresentationType=t.MetaIsoSpatialRepresentationType.vector,
        )

    @property
    def default_name(self):
        return 'WCS'

    ##

    def configure(self):
        super().configure()

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
            raise gws.web.error.NotFound()
        return self.template_response(rd, 'DescribeCoverage', context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_getcoverage(self, rd: ows.Request):
        lcs = self.layer_caps_list_from_request(rd, ['coverageid', 'coverage', 'identifier'])
        if not lcs:
            raise gws.web.error.NotFound()
        return self.render_map_bbox_from_layer_caps_list(lcs, rd)
