import gws
import gws.common.ows.service as ows
import gws.common.ows.service.inspire as inspire
import gws.ext.ows.service.wms as wms


class Config(wms.Config):
    """INSPIRE/WMS Service configuration"""

    pass

class Object(wms.Object):
    def __init__(self):
        super().__init__()
        self.namespaces = gws.extend({}, ows.NAMESPACES, inspire.NAMESPACES)
        self.base_path = gws.dirname(__file__)

    def configure(self):
        super().configure()

    def can_handle(self, req) -> bool:
        return req.kparam('service', '').lower() == 'wms' and req.kparam('inspire', '').lower() == 'true'

    def handle_getcapabilities(self, rd: ows.RequestData):
        return ows.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_tree': ows.layer_node_tree(rd),
            'project_csw_url': self.service_endpoint(rd) + '/service/csw'
        }))
