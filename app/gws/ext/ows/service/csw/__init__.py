import gws

import gws.common.metadata
import gws.common.metadata.inspire
import gws.common.model
import gws.common.ows.service
import gws.common.search.runner
import gws.gis.gml
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.tools.date
import gws.tools.xml2
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows

from . import filter


class Profile(t.Enum):
    ISO = 'ISO'
    DCMI = 'DCMI'


class Config(gws.common.ows.service.Config):
    """CSW Service configuration"""

    profile: Profile = 'ISO'  #: metadata profile


VERSION = '2.0.2'
MAX_LIMIT = 100


class Object(ows.Base):

    def configure(self):
        super().configure()

        self.type = 'csw'
        self.version = VERSION

        self.metas = None
        self.index = None

        self.profile = self.var('profile')
        self.record_template = 'record' + self.profile

        for tpl in 'getCapabilities', 'getRecords', 'getRecordById', 'recordISO', 'recordDCMI':
            self.templates[tpl] = self.configure_template(tpl, 'csw/templates')

        self.templates['describeRecord'] = self.configure_template('describeRecord', 'csw/templates', type='text')

    def configure_metadata(self):
        return gws.extend(
            super().configure_metadata(),
            inspireDegreeOfConformity=t.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=t.MetaInspireKeyword.humanCatalogueViewer,
            inspireResourceType=t.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=t.MetaInspireSpatialDataServiceType.discovery,
            isoScope=t.MetaIsoScope.dataset,
            isoSpatialRepresentationType=t.MetaIsoSpatialRepresentationType.vector,
        )

    def post_configure(self):
        # when using CSW, set meta urls of all objects (that don't have meta.url set) to our service url

        super().post_configure()

        for obj in self.root.find_all():
            meta = gws.get(obj, 'meta')
            if meta and not gws.get(meta, 'url'):
                meta.url = f'{gws.SERVER_ENDPOINT}/cmd/owsHttpGetService/uid/{self.uid}/request/GetRecordById/id/{obj.uid}'

    def handle(self, req) -> t.HttpResponse:
        if self.metas is None:
            self._init_db()

        rd = ows.Request({
            'req': req,
            'project': None,
            'service': self,
        })

        request_param = req.param('request', '')

        if req.method == 'POST':
            try:
                rd.xml = gws.tools.xml2.from_string(req.text_data)
            except gws.tools.xml2.Error:
                raise gws.web.error.BadRequest()
            request_param = rd.xml.name

        return self.dispatch(rd, request_param.lower())

    def handle_getcapabilities(self, rd: ows.Request):
        return self.xml_response(self.render_template(rd, 'getCapabilities'))

    def handle_describerecord(self, rd: ows.Request):
        return self.xml_response(self.render_template(rd, 'describeRecord'))

    def handle_getrecords(self, rd: ows.Request):
        records = self._find_records(rd)

        results = {
            'timestamp': gws.tools.date.now_iso(),
            'next': 0,
            'count_total': len(records),
            'count_return': len(records),
        }

        return self.xml_response(self.render_template(rd, 'getRecords', {
            'record_tags': self._render_records(rd, records),
            'results': results,
        }))

    def handle_getrecordbyid(self, rd: ows.Request):
        meta = self.metas.get(rd.req.param('id'))
        if not meta:
            raise gws.web.error.NotFound()
        tags = self._render_metas(rd, [meta])
        return self.xml_response(self.render_template(rd, 'getRecordById', {
            'record_tags': tags
        }))

    def _init_db(self):
        self.metas = {}
        self.index = []

        for obj in self.root.find_all():
            meta = gws.get(obj, 'meta')
            if not meta:
                continue

            self.metas[obj.uid] = meta

            if gws.get(meta, 'inspireTheme'):
                meta.inspireThemeName = gws.common.metadata.inspire.theme_name(meta.inspireTheme, meta.language)
                meta.inspireThemeDefinition = gws.common.metadata.inspire.theme_definition(meta.inspireTheme, meta.language)

            for f in 'abstract', 'title':
                s = gws.get(meta, f)
                if s:
                    self.index.append((f, s, s.lower(), obj.uid))
            for s in gws.get(meta, 'keywords') or []:
                self.index.append(('subject', s, s.lower(), obj.uid))

    def _find_records(self, rd):
        recs = self.records.values()
        flt = rd.xml.first('Query.Constraint.Filter') if rd.xml else None
        if not flt:
            return recs
        f = filter.Filter(self.index)
        return f.apply(flt.first(), recs)

    def _render_metas(self, rd, metas):
        tags = []

        for m in metas:
            tags.append(self.render_template(rd, self.record_template, {
                'meta': m,
            }, format='tag'))

        return tags
