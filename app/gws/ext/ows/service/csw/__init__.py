import gws

import gws.common.model
import gws.common.ows.service
import gws.common.ows.service.inspire
import gws.common.search.runner
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.gis.gml
import gws.tools.date
import gws.tools.units as units
import gws.common.metadata
import gws.tools.xml2
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows

from . import filter


class Config(gws.common.ows.service.Config):
    """CSW Service configuration"""

    profile: str = 'ISO'  #: ISO or DCMI profile


VERSION = '2.0.2'
MAX_LIMIT = 100


class Object(ows.Base):
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

        self.meta = gws.merge(
            self.meta,
            mandatoryKeyword='humanCatalogueViewer',
            spatialDataServiceType='discovery',
            isoScope='dataset'
        )

    def handle(self, req) -> t.HttpResponse:
        if self.records is None:
            self._load_records()

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
        uid = rd.req.param('id')
        record = self.records.get(gws.as_uid(uid))

        return self.xml_response(self.render_template(rd, 'getRecordById', {
            'record_tags': self._render_records(rd, [record]) if record else [],
        }))

    def _load_records(self):
        metas = self._collect_metadata()

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

    def _collect_metadata(self):
        rs = {}

        for obj in self.find_all():
            meta = getattr(obj, 'meta', None)
            uid = gws.get(meta, 'iso.uid')
            if not uid:
                continue
            m = self._configure_metadata(obj, meta)
            if m:
                rs[gws.as_uid(uid)] = m
        return rs

    def _configure_metadata(self, obj: t.IObject, meta: t.MetaData) -> t.MetaData:
        m = gws.common.metadata.from_config(meta)
        extent = crs = res = None

        if obj.is_a('gws.ext.layer'):
            p: t.ILayer = obj
            extent = p.extent
            crs = p.map.crs
            res = p.resolutions
        else:
            p: t.IProject = obj
            if not obj.is_a('gws.common.project'):
                p = obj.get_closest('gws.common.project')
            if p:
                extent = p.map.extent
                crs = p.map.crs
                res = p.map.resolutions

        if extent:
            m.lonlat_extent = self.lonlat_extent(extent, crs)
            m.proj = gws.gis.proj.as_proj(crs)
            m.resolution = int(min(units.res2scale(r) for r in res))

        if gws.get(m, 'inspire.theme'):
            m.inspire['themeName'] = gws.common.ows.service.inspire.theme_name(m.inspire['theme'], m.language)

        m.iso = gws.merge({
            'spatialType': 'vector',
        }, m.iso)

        m.inspire = gws.merge({
            'qualityExplanation': '',
            'qualityPass': 'false',
            'qualityLineage': '',
        }, m.inspire)

        return m

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
