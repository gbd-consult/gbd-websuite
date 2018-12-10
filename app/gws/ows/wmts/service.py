import gws
import gws.types as t
import gws.tools.xml3
import gws.tools.net

import gws.ows.request
import gws.ows.response

from . import caps


class Service(gws.Object, t.ServiceInterface):
    def __init__(self):
        super().__init__()
        self.type = 'WMTS'

    def configure(self):
        super().configure()

        self.url = self.var('url')

        if self.url:
            xml = gws.ows.request.get_text(
                self.url,
                service='WMTS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            xml = self.var('xml')

        caps.parse(self, xml)
