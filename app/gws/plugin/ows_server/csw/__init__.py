"""CSW service.

Basic implementation of the OGC Catalogue Service for the Web (CSW) standard.
Only a small subset of features is supported.

References:
    - OpenGIS Catalogue Service Implementation Specification 2.0.2 (http://portal.opengeospatial.org/files/?artifact_id=20555)
"""

import gws
import gws.base.ows.server as server
import gws.base.shape
import gws.base.web
import gws.config.util
import gws.lib.extent
import gws.lib.bounds
import gws.lib.xmlx
import gws.lib.datetimex
import gws.lib.crs
import gws.lib.metadata
import gws.lib.mime
import gws.base.search.filter

gws.ext.new.owsService('csw')

_cdir = gws.u.dirname(__file__)

_DEFAULT_TEMPLATES_ISO = [
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/iso/getCapabilities.cx.py',
        subject='ows.GetCapabilities',
        mimeTypes=[gws.lib.mime.XML],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/iso/describeRecord.cx.py',
        subject='ows.DescribeRecord',
        mimeTypes=[gws.lib.mime.XML],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/iso/getRecords.cx.py',
        subject='ows.GetRecords',
        mimeTypes=[gws.lib.mime.XML],
    ),
]

_DEFAULT_METADATA = gws.Metadata(
    name='CSW',
    inspireMandatoryKeyword='infoMapAccessService',
    inspireResourceType='service',
    inspireSpatialDataServiceType='view',
    isoScope='dataset',
    isoServiceFunction='download',
    isoSpatialRepresentationType='vector',
)


class IndexEntry(gws.Data):
    category: str
    text: str
    key: str


class Profile(gws.Enum):
    ISO = 'ISO'
    DCMI = 'DCMI'


class Config(gws.base.ows.server.service.Config):
    """CSW Service configuration"""

    # @TODO no support for DCMI yet
    profile: Profile = Profile.ISO
    """metadata profile"""


class Object(gws.base.ows.server.service.Object):
    protocol = gws.OwsProtocol.CSW
    supportedVersions = ['2.0.2']

    mdMap: dict[str, gws.Metadata]
    index: list[IndexEntry]
    profile: Profile

    def configure(self):
        self.mdMap = {}
        self.profile = Profile.ISO

    def configure_templates(self):
        extra = _DEFAULT_TEMPLATES_ISO
        return gws.config.util.configure_templates_for(self, extra=extra)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.lib.metadata.merge(_DEFAULT_METADATA, self.metadata)

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(
                verb=gws.OwsVerb.GetCapabilities,
                formats=[gws.lib.mime.XML],
                handlerName='handle_get_capabilities',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.DescribeRecord,
                formats=[gws.lib.mime.XML],
                handlerName='handle_describe_record',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetRecords,
                formats=[gws.lib.mime.XML],
                handlerName='handle_get_records',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetRecordById,
                formats=[gws.lib.mime.XML],
                handlerName='handle_get_record_by_id',
            ),
        ]

    def post_configure(self):
        self._collect_metadata()
        self._create_index()
        gws.log.info(f'CSW: configured with {len(self.mdMap)} records')

    ##

    def parse_xml_request(self, xml):
        params = {}

        params['REQUEST'] = xml.name
        return params

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(sr)

    def handle_describe_record(self, sr: server.request.Object):
        return self.template_response(sr)

    def handle_get_records(self, sr: server.request.Object):
        mds = self._find_mds(sr)

        mdc = server.MetadataCollection(
            members=mds,
            numMatched=len(mds),
            numReturned=len(mds),
            timestamp=gws.lib.datetimex.to_iso_string(with_tz=None),
        )

        return self.template_response(
            sr,
            '',
            metadataCollection=mdc,
            next=0,
        )

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

        self.mdMap = {}

        for obj in self.root.find_all():
            md: gws.Metadata = gws.u.get(obj, 'metadata')

            if not md or not md.get('catalogUid'):
                continue

            cid = gws.u.to_uid(md.get('catalogUid'))

            if md.get('metaLinks'):
                gws.log.debug(f'CSW: skip {cid}: has metalinks')
                continue

            if not self.root.app.authMgr.is_public_object(obj):
                gws.log.debug(f'CSW: skip {cid}: not public')
                continue

            md.set('catalogUid', cid)
            md.set('catalogCitationUid', cid)
            md.set('metaLinks', [self._make_link(cid)])

            extent = gws.u.get(obj, 'extent') or gws.u.get(obj, 'map.extent')
            crs = gws.u.get(obj, 'crs') or gws.u.get(obj, 'map.crs')
            if extent and crs:
                md.set('wgsExtent', gws.lib.extent.transform_to_wgs(extent, crs))
                md.set('crs', crs)
                # @TODO get boundingPolygonElement somehow

            self.mdMap[cid] = md

    def _create_index(self):
        self.index = []

        for uid, md in self.mdMap.items():
            ie = IndexEntry(uid=uid)

            if md.title:
                ie.title = md.title
            if md.abstract:
                ie.abstract = md.abstract
            if md.keywords:
                ie.subject = md.keywords
            if md.wgsExtent:
                ie.shape = gws.base.shape.from_extent(md.wgsExtent, gws.lib.crs.WGS84)

            self.index.append(ie)

    def _make_link(self, cid):
        return gws.MetadataLink(
            url=gws.u.action_url_path('owsService', serviceUid=self.uid, request='record', id=cid),
            format=gws.lib.mime.XML,
            type='TC211' if self.profile == 'ISO' else 'DCMI'
        )

    def _find_mds(self, sr: server.request.Object):
        flt_el = None
        if sr.xmlElement:
            flt_el = sr.xmlElement.findfirst('Query/Constraint/Filter')

        if not flt_el:
            return self.mdMap.values()

        flt = gws.base.search.filter.from_fes_element(flt_el)
        m = gws.base.search.filter.Matcher()

        return [
            md
            for md in self.mdMap.values()
            if m.matches(flt, md)
        ]
