import gws
import gws.base.web.error
import gws.lib.date
import gws.lib.extent
import gws.lib.metadata
import gws.lib.mime
import gws.lib.proj
import gws.lib.xml2
import gws.types as t

from . import filter
from .. import core


class Profile(t.Enum):
    ISO = 'ISO'
    DCMI = 'DCMI'


@gws.ext.Config('ows.service.csw')
class Config(core.ServiceConfig):
    """CSW Service configuration"""

    profile: Profile = Profile.ISO  #: metadata profile


@gws.ext.Object('ows.service.csw')
class Object(core.Service):
    protocol = gws.OwsProtocol.CSW
    supported_versions = ['2.0.2']

    metas: t.Dict[str, gws.lib.metadata.Metadata]
    index: t.List
    profile: Profile

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/getRecords.cx',
                subject='ows.GetRecords',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/getRecordById.cx',
                subject='ows.GetRecordById',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/describeRecord.cx',
                subject='ows.DescribeRecord',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/rawRecord.cx',
                subject='ows.RawRecord',
                mimeTypes=['xml'],
            ),
        ]

    ##

    def configure(self):
        self.metas = {}
        self.index = []
        self.profile = t.cast(Profile, self.var('profile'))

    def configure_metadata(self):
        return gws.merge(
            super().configure_metadata(),
            inspireDegreeOfConformity='notEvaluated',
            inspireMandatoryKeyword='humanCatalogueViewer',
            inspireResourceType='service',
            inspireSpatialDataServiceType='discovery',
            isoScope='dataset',
            isoSpatialRepresentationType='vector',
        )

    def post_configure(self):
        self._collect_and_link_metas()
        self._create_index()

    ##

    def handle_request(self, req: gws.IWebRequest) -> gws.ContentResponse:
        rd = core.Request(req=req, project=None, service=self)

        if req.method == 'GET':
            return self.dispatch(rd, req.param('request', 'rawrecord'))

        # CSW should accept POST'ed xml, which can be wrapped in a SOAP envelope

        try:
            rd.xml = gws.lib.xml2.from_string(req.text)
        except gws.lib.xml2.Error:
            raise gws.base.web.error.BadRequest()

        if rd.xml.name.lower() == 'envelope':
            rd.xml_is_soap = True
            try:
                rd.xml = rd.xml.first('body').first()
            except AttributeError:
                raise gws.base.web.error.BadRequest()

        return self.dispatch(rd, rd.xml.name.lower())

    def handle_getcapabilities(self, rd: core.Request):
        return self.template_response(rd, gws.OwsVerb.GetCapabilities, context={
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_describerecord(self, rd: core.Request):
        return self.template_response(rd, gws.OwsVerb.DescribeRecord, context={
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_getrecords(self, rd: core.Request):
        metas = self._find_metas(rd)

        results = {
            'timestamp': gws.lib.date.now_iso(with_tz=False),
            'next': 0,
            'count_total': len(metas),
            'count_return': len(metas),
        }

        return self.template_response(rd, gws.OwsVerb.GetRecords, context={
            'metas': [m.values for m in metas],
            'results': results,
            'with_soap': rd.xml_is_soap,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_getrecordbyid(self, rd: core.Request):
        meta = self.metas.get(rd.req.param('id'))
        if not meta:
            raise gws.base.web.error.NotFound()

        return self.template_response(rd, gws.OwsVerb.GetRecordById, context={
            'meta': meta.values,
            'with_soap': rd.xml_is_soap,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_rawrecord(self, rd: core.Request):
        # RawRecord is our internal method to return bare metadata without the GetRecordByIdResponse envelope
        meta = self.metas.get(rd.req.param('id'))
        if not meta:
            raise gws.base.web.error.NotFound()

        return self.template_response(rd, t.cast(gws.OwsVerb, 'RawRecord'), context={
            'meta': meta.values,
            'with_soap': False,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    ##

    def _collect_and_link_metas(self):
        self.metas = {}

        # collect objects whose metadata should be published in the catalog
        #
        # - object should have `metadata`
        # - object must be public
        # - `metadata` should have `catalogUid`
        # - `metadata.metaLinks` should be empty
        #
        # `metadata.metaLinks` will be set to our csw url

        for obj in self.root.find_all():
            md: gws.lib.metadata.Metadata = gws.get(obj, 'metadata')

            if not md:
                continue

            if not gws.is_public_object(obj):
                continue

            cid = md.get('catalogUid')
            if not cid:
                continue

            cid = gws.to_uid(cid)

            if md.get('metaLinks'):
                continue

            csw_link = gws.lib.metadata.Link(
                url=gws.action_url_path('owsService', serviceUid=self.uid, id=cid),
                formatName=gws.lib.mime.XML,
                type='TC211' if self.profile == 'ISO' else 'DCMI'
            )

            ext = {
                'catalogCitationUid': cid,
                'metaLinks': [csw_link],
            }

            extent = gws.get(obj, 'extent') or gws.get(obj, 'map.extent')
            crs = gws.get(obj, 'crs') or gws.get(obj, 'map.crs')
            if extent and crs:
                ext['extent4326'] = gws.lib.extent.transform_to_4326(extent, crs)
                ext['proj'] = gws.lib.proj.to_proj(crs)

            md.extend(ext)
            self.metas[cid] = md

    def _create_index(self):
        self.index = []

        for uid, md in self.metas.items():
            s = gws.get(md, 'title')
            if s:
                self.index.append(['title', s, s.lower(), uid])
            s = gws.get(md, 'abstract')
            if s:
                self.index.append(['abstract', s, s.lower(), uid])
            s = gws.get(md, 'keywords')
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
