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


class LinkConfig(t.Config):
    type: str
    href: str


class RecordConfig(t.Config):
    """Describes a Record in the CSW catalog"""

    uid: str
    meta: t.Optional[t.MetaData]  #: record meta data
    extent: t.Optional[t.Extent]
    links: t.Optional[t.List[LinkConfig]]
    crs: t.Optional[t.crsref]  #: record CRS


class Config(gws.common.ows.service.Config):
    """CSW Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates
    meta: t.Optional[t.MetaData]  #: service meta data
    crs: t.crsref  #: default CRS for all records
    records: t.List[RecordConfig]


VERSION = '2.0.2'
MAX_LIMIT = 100


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.service_class = 'csw'
        self.service_type = 'csw'
        self.version = VERSION

        self.namespaces = gws.extend({}, ows.NAMESPACES, inspire.NAMESPACES)

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'getRecords', 'getRecordById', 'record':
            self.templates[tpl] = self.configure_template(tpl, 'csw/templates')

        self.templates['describeRecord'] = self.configure_template('describeRecord', 'csw/templates', type='text')

        self.meta = gws.common.metadata.read(self.var('meta'))
        self.crs = self.var('crs')

        self.records = [self._prepare_record(r) for r in self.var('records')]

    def can_handle(self, req) -> bool:
        if req.is_get:
            return req.kparam('srv', '').lower() == self.service_class
        if req.is_post:
            # @TODO
            return 'csw' in req.post_data

    def handle(self, req) -> t.HttpResponse:
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
        return ows.xml_response(self.render_template(rd, 'getCapabilities', {
            'service_meta': self.meta,
        }))

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

        tags, used_namespaces = self._render_records(rd, records)

        return ows.xml_response(rd.service.render_template(rd, 'getRecords', {
            'record_tags': tags,
            'results': results,
            'service_meta': self.meta,
            'used_namespaces': used_namespaces,
        }))

    def handle_getrecordbyid(self, rd: ows.RequestData):
        id = rd.req.kparam('id')
        records = [r for r in self.records if r.uid == id]

        tags, used_namespaces = self._render_records(rd, records)

        return ows.xml_response(rd.service.render_template(rd, 'getRecordById', {
            'record_tags': tags,
            'service_meta': self.meta,
            'used_namespaces': used_namespaces,
        }))

    def _prepare_record(self, rec):
        rec.meta.language = rec.meta.language or self.meta.language
        rec.meta.language3 = rec.meta.language3 or self.meta.language3
        rec.crs = rec.crs or self.crs

        if rec.extent:
            rec.lonlat_extent = ows.lonlat_extent(rec.extent, rec.crs)

        for s in rec.links:
            if s.type.lower() in ('wms', 'wfs'):
                s.scheme = 'OGC:' + s.type.upper()

        return rec

    def _find_records(self, rd):
        # @TODO filters
        return self.records

    def _render_records(self, rd, records):
        tags = []
        used_namespaces = set()

        for record in records:
            ns, tag = self.render_template(rd, 'record', {
                'service_meta': self.meta,
                'record': record,
            }, format='tag')
            used_namespaces.update(ns)
            tags.append(tag)

        return tags, used_namespaces
