import gws

import gws.common.datamodel
import gws.common.ows.service
import gws.common.search.runner
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.ows.gml
import gws.tools.date
import gws.common.metadata
import gws.tools.xml3
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows
import gws.common.ows.service.inspire as inspire


class TemplatesConfig(t.Config):
    """CSW service templates"""

    getCapabilities: t.Optional[t.TemplateConfig]  #: xml template for the CSW capabilities document
    getRecords: t.Optional[t.TemplateConfig]  #: xml template for the CSW getRecords document


class Config(gws.common.ows.service.Config):
    """CSW Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates


VERSION = '2.0.2'
MAX_LIMIT = 100


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.service_class = 'csw'
        self.service_type = 'csw'
        self.version = VERSION

        self.metas = None

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'getRecords', 'getRecordById', 'record':
            self.templates[tpl] = self.configure_template(tpl, 'csw/templates')

        self.templates['describeRecord'] = self.configure_template('describeRecord', 'csw/templates', type='text')

        self.meta.inspire = gws.extend({
            'mandatoryKeyword': 'humanCatalogueViewer',
            'spatialDataServiceType': 'discovery',
        }, self.meta.inspire)

        self.meta.iso = gws.extend({
            'scope': 'dataset',
        }, self.meta.iso)

    def can_handle(self, req) -> bool:
        if req.is_get:
            return req.kparam('srv', '').lower() == self.service_class
        if req.is_post:
            # @TODO
            return 'csw' in req.post_data

    def handle(self, req) -> t.HttpResponse:
        if self.metas is None:
            self.metas = ows.collect_metadata(self)

        rd = ows.RequestData({
            'req': req,
            'project': None,
            'service': self,
        })

        request_param = req.kparam('request', '')

        if req.is_post:
            try:
                rd.xml = gws.tools.xml3.from_string(req.post_data)
            except gws.tools.xml3.Error:
                raise gws.web.error.BadRequest()
            request_param = rd.xml.name

        return self.dispatch(rd, request_param.lower())

    def handle_getcapabilities(self, rd: ows.RequestData):
        return ows.xml_response(self.render_template(rd, 'getCapabilities', {}))

    def handle_describerecord(self, rd: ows.RequestData):
        return ows.xml_response(self.render_template(rd, 'describeRecord', {}))

    def handle_getrecords(self, rd: ows.RequestData):
        records = self._find_records(rd)

        results = {
            'timestamp': gws.tools.date.now_iso(),
            'next': 0,
            'count_total': len(records),
            'count_return': len(records),
        }

        return ows.xml_response(rd.service.render_template(rd, 'getRecords', {
            'record_tags': self._render_records(rd, records),
            'results': results,
        }))

    def handle_getrecordbyid(self, rd: ows.RequestData):
        record = self.metas.get(rd.req.kparam('id'))
        if not record:
            raise gws.web.error.NotFound()

        return ows.xml_response(rd.service.render_template(rd, 'getRecordById', {
            'record_tags': self._render_records(rd, [record]),
        }))

    def _find_records(self, rd):
        # @TODO filters
        return self.metas.values()

    def _render_records(self, rd, records):
        tags = []

        for record in records:
            tags.append(self.render_template(rd, 'record', {
                'meta': record,
            }, format='tag'))

        return tags
