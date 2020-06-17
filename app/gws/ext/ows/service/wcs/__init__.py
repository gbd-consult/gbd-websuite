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
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows


class Config(gws.common.ows.service.Config):
    """WCS Service configuration"""
    pass


VERSION = '2.0.1'


class Object(ows.Base):

    @property
    def service_link(self):
        return t.MetaLink(
            url=self.url,
            scheme='OGC:WCS',
            function='search'
        )

    def configure(self):
        super().configure()

        self.type = 'wcs'
        self.version = VERSION

        for tpl in 'getCapabilities', 'describeCoverage':
            self.templates[tpl] = self.configure_template(tpl, 'wcs/templates/')


    def configure_metadata(self):
        return gws.extend(
            super().configure_metadata(),
            inspireDegreeOfConformity=t.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=t.MetaInspireKeyword.infoMapAccessService,
            inspireResourceType=t.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=t.MetaInspireSpatialDataServiceType.view,
            isoScope=t.MetaIsoScope.dataset,
            isoSpatialRepresentationType=t.MetaIsoSpatialRepresentationType.vector,
        )

    def handle_getcapabilities(self, rd: ows.Request):
        nodes = self.layer_node_list(rd)
        return self.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_list': nodes,
        }))
