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
        self.namespaces = gws.extend({}, ows.NAMESPACES, inspire.NAMESPACES)
        self.metas = None

    def configure(self):
        super().configure()

        for tpl in ('generic',):
            self.templates[tpl] = self.configure_template(tpl, 'isometa/templates')

    def handle(self, req) -> t.HttpResponse:
        if self.metas is None:
            self.metas = _collect_metadata(self)

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

def _collect_metadata(service):
    rs = {}

    for obj in service.find_all():
        meta = getattr(obj, 'meta', None)
        if not meta or not meta.get('isoUid'):
            continue
        m = _configure_metadata(obj)
        if m:
            rs[m.isoUid] = m
    return rs


def _configure_metadata(obj: t.ObjectInterface):
    m: t.MetaData = gws.common.metadata.read(obj.meta)

    la = None
    if obj.is_a('gws.common.project'):
        la = obj.map
    elif obj.is_a('gws.ext.layer'):
        la = obj

    if la:
        m.proj = gws.gis.proj.as_proj(la.crs)
        m.lonlat_extent = ows.lonlat_extent(la.extent, la.crs)
        m.resolution = int(min(gws.tools.misc.res2scale(r) for r in la.resolutions))

    if m.inspireTheme:
        m.inspireThemeName = inspire.theme_name(m.inspireTheme, m.language)

    m.isoProps = gws.extend({
        'spatialType': 'vector',
    }, m.isoProps)

    m.inspireProps = gws.extend({
        'qualityExplanation': '',
        'qualityPass': 'false',
        'qualityLineage': '',
    }, m.inspireProps)

    return m
