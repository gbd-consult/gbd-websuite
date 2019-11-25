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

from . import filter


class Config(gws.common.ows.service.Config):
    """CSW Service configuration"""

    profile: str = 'ISO'  #: ISO or DCMI profile


VERSION = '2.0.2'
MAX_LIMIT = 100


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.type = 'csw'
        self.version = VERSION

        self.records = None
        self.index = None

        self.record_template = ''
        self.profile = ''

    def configure(self):
        super().configure()

        self.profile = self.var('profile')
        self.record_template = 'record' + self.profile

        for tpl in 'getCapabilities', 'getRecords', 'getRecordById', 'recordISO', 'recordDCMI':
            self.templates[tpl] = self.configure_template(tpl, 'csw/templates')

        self.templates['describeRecord'] = self.configure_template('describeRecord', 'csw/templates', type='text')

        self.meta.inspire = gws.extend({
            'mandatoryKeyword': 'humanCatalogueViewer',
            'spatialDataServiceType': 'discovery',
        }, self.meta.inspire)

        self.meta.iso = gws.extend({
            'scope': 'dataset',
        }, self.meta.iso)

    def handle(self, req) -> t.HttpResponse:
        if self.records is None:
            self._load_records()

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
        uid = rd.req.kparam('id')
        record = self.records.get(gws.as_uid(uid))

        return ows.xml_response(rd.service.render_template(rd, 'getRecordById', {
            'record_tags': self._render_records(rd, [record]) if record else [],
        }))

    def _load_records(self):
        metas = ows.collect_iso_metadata(self)

        self.records = {}
        self.index = []

        for uid, meta in metas.items():
            rec = t.Data(gws.as_dict(meta))
            rec.index = len(self.records) + 1
            self.records[uid] = rec

        for rec in self.records.values():
            for f in 'abstract', 'title':
                s = rec.get(f)
                if s:
                    self.index.append((f, s, s.lower(), rec.index))
            for s in rec.keywords:
                self.index.append(('subject', s, s.lower(), rec.index))


    def _find_records(self, rd):
        recs = self.records.values()
        flt = rd.xml.first('Query.Constraint.Filter') if rd.xml else None
        if not flt:
            return recs
        f = filter.Filter(self.index)
        return f.apply(flt.first(), recs)

    def _render_records(self, rd, records):
        tags = []

        for record in records:
            tags.append(self.render_template(rd, self.record_template, {
                'meta': record,
            }, format='tag'))

        return tags
