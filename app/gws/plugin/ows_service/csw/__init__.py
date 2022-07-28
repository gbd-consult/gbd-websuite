import gws
import gws.base.web.error
import gws.lib.date
import gws.gis.extent
import gws.lib.metadata
import gws.lib.mime
import gws.lib.xml2 as xml2
import gws.types as t

from . import filter
from .. import core


class Profile(t.Enum):
    ISO = 'ISO'
    DCMI = 'DCMI'


@gws.ext.config.owsService('csw')
class Config(core.ServiceConfig):
    """CSW Service configuration"""
    # @TODO no support for DCMI yet
    # profile: Profile = Profile.ISO  #: metadata profile
    pass


@gws.ext.object.owsService('csw')
class Object(core.Service):
    protocol = gws.OwsProtocol.CSW
    supported_versions = ['2.0.2']

    records: t.Dict[str, gws.IMetadata]
    index: t.List
    profile: Profile

    @property
    def default_templates(self):
        base = gws.dirname(__file__) + '/templates'
        return [
            gws.Config(
                type='py',
                path=f'{base}/getCapabilities.py',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
                access='all:allow',
            ),
            gws.Config(
                type='py',
                path=f'{base}/getRecords.py',
                subject='ows.getRecords',
                mimeTypes=['xml'],
                access='all:allow',
            ),
            gws.Config(
                type='py',
                path=f'{base}/getRecordById.py',
                subject='ows.GetRecordById',
                mimeTypes=['xml'],
                access='all:allow',
            ),
            gws.Config(
                type='py',
                path=f'{base}/record.py',
                subject='ows.Record',
                mimeTypes=['xml'],
                access='all:allow',
            ),
        ]

    @property
    def default_metadata(self):
        return gws.Data(
            inspireDegreeOfConformity='notEvaluated',
            inspireMandatoryKeyword='humanCatalogueViewer',
            inspireResourceType='service',
            inspireSpatialDataServiceType='discovery',
            isoScope='dataset',
            isoSpatialRepresentationType='vector',
        )

    ##

    def configure(self):
        self.records = {}
        self.index = []
        self.profile = Profile.ISO

    def post_configure(self):
        self._collect_metadata()
        self._create_index()
        cnt = len(self.records)
        gws.log.info(f'CSW service configured with {cnt} records')

    ##

    def handle_request(self, req: gws.IWebRequest) -> gws.ContentResponse:
        rd = core.Request(req=req, project=None, service=self)

        if req.method == 'GET':
            return self.dispatch_request(rd, req.param('request', default='record'))

        # CSW should accept POST'ed xml, which can be wrapped in a SOAP envelope

        try:
            rd.xml_element = xml2.from_string(req.text)
        except xml2.Error:
            raise gws.base.web.error.BadRequest()

        if rd.xml_element.name.lower() == 'envelope':
            rd.xml_is_soap = True
            try:
                rd.xml_element = xml2.first(xml2.first('body'))
            except Exception:
                raise gws.base.web.error.BadRequest()

        return self.dispatch_request(rd, xml2.unqualify_name(rd.xml_element.name.lower()))

    def handle_getcapabilities(self, rd: core.Request):
        return self.template_response(rd, gws.OwsVerb.GetCapabilities, context={
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_describerecord(self, rd: core.Request):
        xml = gws.read_file(gws.dirname(__file__) + '/templates/describeRecord.xml')
        return gws.ContentResponse(content=xml, mime=gws.lib.mime.XML)

    def handle_getrecords(self, rd: core.Request):
        records = self._find_records(rd)

        results = {
            'timestamp': gws.lib.date.now_iso(with_tz=False),
            'next': 0,
            'count_total': len(records),
            'count_return': len(records),
        }

        return self.template_response(rd, gws.OwsVerb.GetRecords, context={
            'records': [md.values for md in records],
            'results': results,
            'with_soap': rd.xml_is_soap,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_getrecordbyid(self, rd: core.Request):
        md = self.records.get(rd.req.param('id'))
        if not md:
            raise gws.base.web.error.NotFound()

        return self.template_response(rd, gws.OwsVerb.GetRecordById, context={
            'record': md.values,
            'with_soap': rd.xml_is_soap,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    def handle_record(self, rd: core.Request):
        # Record is our internal method to return bare metadata without the GetRecordByIdResponse envelope
        # see _make_link below

        md = self.records.get(rd.req.param('id'))
        if not md:
            raise gws.base.web.error.NotFound()

        return self.template_response(rd, t.cast(gws.OwsVerb, 'Record'), context={
            'record': md.values,
            'with_soap': False,
            'profile': self.profile,
            'version': self.request_version(rd),
        })

    ##

    def _collect_metadata(self):
        # collect objects whose metadata should be published in the catalog
        #
        # - object should have `metadata`
        # - object must be public
        # - `metadata` should have `catalogUid`
        # - `metadata.metaLinks` should be empty
        #
        # `metadata.metaLinks[0]` will be set to our csw url

        self.records = {}

        for obj in self.root.find_all():
            md: gws.lib.metadata.Metadata = gws.get(obj, 'metadata')

            if not md or not md.get('catalogUid'):
                continue

            cid = gws.to_uid(md.get('catalogUid'))

            if md.get('metaLinks'):
                gws.log.debug(f'csw: skip {cid}: has metalinks')
                continue

            if not gws.is_public_object(obj):
                gws.log.debug(f'csw: skip {cid}: not public')
                continue

            md.set('catalogUid', cid)
            md.set('catalogCitationUid', cid)
            md.set('metaLinks', [self._make_link(cid)])

            extent = gws.get(obj, 'extent') or gws.get(obj, 'map.extent')
            crs = gws.get(obj, 'crs') or gws.get(obj, 'map.crs')
            if extent and crs:
                md.set('extent4326', gws.gis.extent.transform_to_4326(extent, crs))
                md.set('crs', crs)
                # @TODO get boundingPolygonElement somehow

            self.records[cid] = md

    def _make_link(self, cid):
        return gws.MetadataLink(
            url=gws.action_url_path('owsService', serviceUid=self.uid, request='record', id=cid),
            formatName=gws.lib.mime.XML,
            type='TC211' if self.profile == 'ISO' else 'DCMI'
        )

    def _create_index(self):
        self.index = []

        for uid, md in self.records.items():
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

    def _find_records(self, rd: core.Request):
        flt = None
        if rd.xml_element:
            flt = xml2.first(xml2.first('Query.Constraint.Filter'))
        if not flt:
            return self.records.values()
        f = filter.Filter(self.index)
        return f.apply(flt, self.records.values())
