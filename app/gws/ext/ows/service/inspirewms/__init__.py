import gws
import gws.common.ows.service as ows
import gws.common.ows.service.inspire as inspire
import gws.ext.ows.service.wms as wms
import gws.ext.ows.service.inspirewfs as inspirewfs


class Config(wms.Config):
    """INSPIRE/WMS Service configuration"""

    pass


class Object(wms.Object):
    def __init__(self):
        super().__init__()

        self.service_class = 'inspirewms'
        self.service_type = 'wms'
        self.version = wms.VERSION

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'getFeatureInfo':
            self.templates[tpl] = self.configure_template(tpl, 'inspirewms/templates')
        inspirewfs.configure_inspire_templates(self)

    def handle_getfeatureinfo(self, rd: ows.RequestData):
        features = wms.find_features(rd)
        return inspirewfs.render_inspire_features(rd, features, 'getFeatureInfo')
