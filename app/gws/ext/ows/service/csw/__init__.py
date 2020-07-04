import gws

import gws.common.metadata
import gws.common.metadata.inspire
import gws.common.model
import gws.common.ows.service
import gws.common.search.runner
import gws.gis.gml
import gws.gis.extent
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

        self.templates['getCapabilities'] = self.configure_template('getCapabilities', 'csw/templates/')
        self.templates['getRecords'] = self.configure_template('getRecords', f'csw/templates/getRecords_{self.profile}.cx')
        self.templates['getRecordById'] = self.configure_template('getRecordById', f'csw/templates/getRecordById_{self.profile}.cx')
        self.templates['describeRecord'] = self.configure_template('describeRecord', 'csw/templates/', type='text')

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

        self.metas = {}

        for obj in self.root.find_all():
            meta: t.MetaData = gws.get(obj, 'meta')
            if not meta:
                continue
            if not meta.catalogUid:
                if meta.authorityIdentifier:
                    meta.catalogUid = meta.authorityIdentifier
                else:
                    meta.catalogUid = obj.uid
            if not meta.url:
                meta.url = f'{gws.SERVER_ENDPOINT}/cmd/owsHttpService/uid/{self.uid}/request/GetRecordById/id/{obj.uid}'

            extent = gws.get(obj, 'extent') or gws.get(obj, 'map.extent')
            crs = gws.get(obj, 'crs') or gws.get(obj, 'map.crs')
            if extent and crs:
                meta.extent4326 = gws.gis.extent.transform_to_4326(extent, crs)

            self.metas[obj.uid] = meta

        self._create_index()

    def handle(self, req) -> t.HttpResponse:
        rd = ows.Request(req=req, project=None, service=self)

        if req.method == 'GET':
            return self.dispatch(rd, req.param('request', ''))

        # CSW should accept POST'ed xml, which can be wrapped in a SOAP envelope

        try:
            rd.xml = gws.tools.xml2.from_string(req.text)
        except gws.tools.xml2.Error:
            raise gws.web.error.BadRequest()

        if rd.xml.name.lower() == 'envelope':
            rd.xml_is_soap = True
            try:
                rd.xml = rd.xml.first('body').first()
            except AttributeError:
                raise gws.web.error.BadRequest()

        return self.dispatch(rd, rd.xml.name.lower())

    def handle_getcapabilities(self, rd: ows.Request):
        return self.xml_response(self.render_template(rd, 'getCapabilities'))

    def handle_describerecord(self, rd: ows.Request):
        return self.xml_response(self.render_template(rd, 'describeRecord'))

    def handle_getrecords(self, rd: ows.Request):
        metas = self._find_metas(rd)

        results = {
            'timestamp': gws.tools.date.now_iso(with_tz=False),
            'next': 0,
            'count_total': len(metas),
            'count_return': len(metas),
        }

        return self.xml_response(self.render_template(rd, 'getRecords', {
            'metas': metas,
            'results': results,
            'with_soap': rd.xml_is_soap,
        }))

    def handle_getrecordbyid(self, rd: ows.Request):
        meta = self.metas.get(rd.req.param('id'))
        if not meta:
            raise gws.web.error.NotFound()
        return self.xml_response(self.render_template(rd, 'getRecordById', {
            'meta': meta,
            'with_soap': rd.xml_is_soap,
        }))

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
