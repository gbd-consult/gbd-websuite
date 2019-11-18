"""ISO19115 metadata"""

import gws
import gws.web.error
import gws.tools.misc
import gws.common.metadata
import gws.gis.proj

import gws.types as t

import gws.common.ows.service as ows
import gws.common.ows.service.inspire as inspire


class TemplatesConfig(t.Config):
    generic: t.Optional[t.TemplateConfig]  #: generic metadata template


class Config(ows.Config):
    """Metadata Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.service_class = 'isometa'
        self.service_type = 'isometa'
        self.version = '1.0'
        self.metas = None

    def configure(self):
        super().configure()

        for tpl in ('generic',):
            self.templates[tpl] = self.configure_template(tpl, 'isometa/templates')

    def handle(self, req) -> t.HttpResponse:
        if self.metas is None:
            self.metas = ows.collect_metadata(self)

        meta = self.metas.get(req.kparam('id'))
        if not meta:
            raise gws.web.error.NotFound()

        rd = ows.RequestData({
            'req': req,
            'project': None,
            'service': self,
        })

        return ows.xml_response(self.render_template(rd, 'generic', {
            'mw': meta,
        }))

