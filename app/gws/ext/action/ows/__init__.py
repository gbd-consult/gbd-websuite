import os

import gws
import gws.web.error
import gws.types as t

from . import wms


class TemplatesConfig:
    wmsGetCapabilities: t.Optional[t.TemplateConfig]  ## template for the WMS capabilities document


class Config(t.WithTypeAndAccess):
    """OWS server action"""

    templates: t.Optional[TemplatesConfig]  ## OWS templates


class Object(gws.ActionObject):
    @property
    def props(self):
        # no client props for this action
        return None

    def configure(self):
        super().configure()

        self.templates = t.Data()

        self.templates.wmsGetCapabilities = self.init_template('wmsGetCapabilities')

    def init_template(self, name):
        p = self.var('templates.' + name)
        if p:
            return self.create_object('gws.ext.template', p)

        return self.create_shared_object(
            'gws.ext.template',
            'generic_' + name,
            {
                'type': 'html',
                'path': os.path.dirname(__file__) + f'/templates/{name}.cx'
            })

    def http_get(self, req, p) -> t.HttpResponse:
        ps = {k.lower(): v for k, v in req.params.items()}
        service = ps.get('service', '').lower()
        if service == 'wms':
            return wms.request(self, req, ps)
        raise gws.web.error.NotFound()
