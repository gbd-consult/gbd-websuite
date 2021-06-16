import gws

import gws.base.metadata
import gws.base.metadata.inspire
import gws.base.model
import gws.base.ows.service
import gws.base.search.runner
import gws.gis.gml
import gws.gis.extent
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.lib.date
import gws.lib.xml2
import gws.lib.mime
import gws.web.error

import gws.types as t

import gws.base.ows.service as ows

from . import filter


class Profile(t.Enum):
    ISO = 'ISO'
    DCMI = 'DCMI'


class Config(gws.base.ows.service.Config):
    """CSW Service configuration"""

    profile: Profile = 'ISO'  #: metadata profile


class Object(ows.Base):

    @property
    def default_templates(self):
        base = gws.APP_DIR + '/gws/ext/ows/service/csw/templates'

        return [
            t.Config(
                type='xml',
                path=base + '/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=base + '/getRecords.cx',
                subject='ows.GetRecords',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=base + '/getRecordById.cx',
                subject='ows.GetRecordById',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=base + '/describeRecord.cx',
                subject='ows.DescribeRecord',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=base + '/rawRecord.cx',
                subject='ows.RawRecord',
                mimeTypes=['xml'],
            ),
        ]

    @property
    def default_name(self):
        return 'CSW'

    ##

    def configure(self):
        super().configure()

        self.type = 'csw'
        self.supported_versions = ['2.0.2']

        self.metas = None
        self.index = None

        self.profile = self.var('profile')

    def configure_metadata(self):
        return gws.extend(
            super().configure_metadata(),
            inspireDegreeOfConformity=t.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=t.MetaInspireMandatoryKeyword.humanCatalogueViewer,
            inspireResourceType=t.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=t.MetaInspireSpatialDataServiceType.discovery,
            isoScope=t.MetaIsoScope.dataset,
            isoSpatialRepresentationType=t.MetaIsoSpatialRepresentationType.vector,
        )

    def post_configure(self):
        super().post_configure()

        self.metas = {}

        for obj in self.root.find_all():
            meta: t.MetaData = gws.get(obj, 'meta')
            if not meta:
                continue
            if not meta.catalogUid:
                meta.catalogUid = meta.authorityIdentifier or obj.uid
                meta.catalogCitationUid = meta.catalogUid
            if not meta.url:
                # when using CSW, set meta urls of all objects (that don't have meta.url set) to our "raw record" url
                meta.url = f'{gws.SERVER_ENDPOINT}/cmd/owsHttpService/uid/{self.uid}/id/{obj.uid}'
                meta.urlType = "TC211" if self.profile == 'ISO' else 'DCMI'  ## @TODO
                meta.urlFormat = "text/xml"
            elif meta.url == 'SKIP':
                meta.url = None

            extent = gws.get(obj, 'extent') or gws.get(obj, 'map.extent')
            crs = gws.get(obj, 'crs') or gws.get(obj, 'map.crs')
            if extent and crs:
                meta.extent4326 = gws.gis.extent.transform_to_4326(extent, crs)
                meta.proj = gws.gis.proj.as_proj(crs)

            self.metas[obj.uid] = meta

        self._create_index()

    ##

    def handle(self, req) -> t.HttpResponse:
        rd = ows.Request(req=req, project=None, service=self)

        if req.method == 'GET':
            return self.dispatch(rd, req.param('request', '') or 'rawrecord')

        # CSW should accept POST'ed xml, which can be wrapped in a SOAP envelope

        try:
            rd.xml = gws.lib.xml2.from_string(req.text)
        except gws.lib.xml2.Error:
            raise gws.web.error.BadRequest()

        if rd.xml.name.lower() == 'envelope':
            rd.xml_is_soap = True
            try:
                rd.xml = rd.xml.first('body').first()
            except AttributeError:
                raise gws.web.error.BadRequest()

        return self.dispatch(rd, rd.xml.name.lower())

    def handle_getcapabilities(self, rd: ows.Request):
        return self.template_response(rd, 'GetCapabilities', context={
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_describerecord(self, rd: ows.Request):
        return self.template_response(rd, 'DescribeRecord', context={
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_getrecords(self, rd: ows.Request):
        metas = self._find_metas(rd)

        results = {
            'timestamp': gws.lib.date.now_iso(with_tz=False),
            'next': 0,
            'count_total': len(metas),
            'count_return': len(metas),
        }

        records = []

        for meta in metas:
            rec = self.render_template(rd, 'RawRecord', format='tag', context={
                'meta': meta,
                'profile': self.profile,
                'version': self.request_version(rd),
            })
            records.append(rec.content)

        return self.template_response(rd, 'GetRecords', context={
            'records': records,
            'results': results,
            'with_soap': rd.xml_is_soap,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_getrecordbyid(self, rd: ows.Request):
        meta = self.metas.get(rd.req.param('id'))
        if not meta:
            raise gws.web.error.NotFound()

        rec = self.render_template(rd, 'RawRecord', format='tag', context={
            'meta': meta,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

        return self.template_response(rd, 'GetRecordById', context={
            'record': rec.content,
            'with_soap': rd.xml_is_soap,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_rawrecord(self, rd: ows.Request):
        # RawRecord is our internal method to return bare metadata without the GetRecordByIdResponse envelope
        meta = self.metas.get(rd.req.param('id'))
        if not meta:
            raise gws.web.error.NotFound()
        return self.template_response(rd, 'RawRecord', context={
            'meta': meta,
            'with_soap': False,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    ##

    def _collect_metas(self):
        self.metas = {}

        for obj in self.root.find_all():
            meta = gws.get(obj, 'meta')
            if not meta or not meta.catalogUid:
                continue
            self.metas[obj.uid] = meta

    def _create_index(self):
        self.index = []

        for uid, meta in self.metas.items():
            s = gws.get(meta, 'title')
            if s:
                self.index.append(['title', s, s.lower(), uid])
            s = gws.get(meta, 'abstract')
            if s:
                self.index.append(['abstract', s, s.lower(), uid])
            s = gws.get(meta, 'keywords')
            if s:
                for kw in s:
                    self.index.append(('subject', kw, kw.lower(), uid))

    def _find_metas(self, rd):
        recs = self.metas.values()
        flt = rd.xml.first('Query.Constraint.Filter') if rd.xml else None
        if not flt:
            return recs
        f = filter.Filter(self.index)
        return f.apply(flt.first(), recs)
