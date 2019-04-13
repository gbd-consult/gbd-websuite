import os

import gws
import gws.web.error
import gws.types as t

from . import wms


class Config(t.WithTypeAndAccess):
    """OWS server action"""
    pass


class Object(gws.Object):

    def configure(self):
        super().configure()

        p = self.var('templates.wms.GetCapabilities')
        if p:
            self.wms_caps_template = self.create_object('gws.ext.template', p)
        else:
            self.wms_caps_template = self.create_shared_object(
                'gws.ext.template',
                'wms_caps_template',
                {
                    'type': 'html',
                    'path': os.path.dirname(__file__) + '/templates/wms_getcapabilities.cx.xml'
                })

    def http_get(self, req, p) -> t.HttpResponse:
        ps = {k.lower(): v for k, v in req.params.items()}
        service = ps.get('service', '').lower()
        if service == 'wms':
            return wms.request(self, req, ps)
        raise gws.web.error.NotFound()
