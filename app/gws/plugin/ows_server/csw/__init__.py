"""CSW service.

Basic implementation of the OGC Catalogue Service for the Web (CSW) standard.
Only a small subset of features is supported.

References:
    - OpenGIS Catalogue Service Implementation Specification 2.0.2 (http://portal.opengeospatial.org/files/?artifact_id=20555)
"""

from typing import cast
import gws
import gws.base.metadata
import gws.base.map
import gws.base.ows.server as server
import gws.base.search.filter
import gws.base.shape
import gws.config.util
import gws.lib.crs
import gws.lib.datetimex
import gws.lib.extent
import gws.lib.mime
import gws.lib.uom

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
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/iso/getRecords.cx.py',
        subject='ows.GetRecordById',
        mimeTypes=[gws.lib.mime.XML],
    ),
]

_DEFAULT_METADATA = dict(
    name='CSW',
    inspireMandatoryKeyword='infoMapAccessService',
    inspireDegreeOfConformity='notEvaluated',
    inspireResourceType='service',
    inspireSpatialDataServiceType='view',
    isoScope='dataset',
    isoServiceFunction='download',
    isoSpatialRepresentationType='vector',
)


class Profile(gws.Enum):
    """Metadata profile for CSW service."""

    ISO = 'ISO'
    """ISO 19115 metadata profile."""
    DCMI = 'DCMI'
    """Dublin Core metadata profile."""


class Config(server.service.Config):
    """CSW Service configuration"""

    profile: Profile = Profile.ISO
    """Metadata profile."""


class Object(server.service.Object):
    protocol = gws.OwsProtocol.CSW
    supportedVersions = ['2.0.2']

    mdMap: dict[str, gws.Metadata]
    profile: Profile

    def configure(self):
        self.mdMap = {}
        self.profile = Profile.ISO

    def configure_templates(self):
        extra = _DEFAULT_TEMPLATES_ISO
        return gws.config.util.configure_templates_for(self, extra=extra)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.base.metadata.from_args(_DEFAULT_METADATA, self.metadata)

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
        self.collect_metadata()

    ##

    def parse_xml_request(self, xml):
        params = {}

        params['REQUEST'] = xml.name
        return params

    ##

    def init_request(self, req):
        sr = super().init_request(req)
        sr.load_project()
        return sr

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(sr)

    def handle_describe_record(self, sr: server.request.Object):
        return self.template_response(sr)

    def handle_get_records(self, sr: server.request.Object):
        mds = self._find_metas(sr)

        mdc = server.MetadataCollection(
            members=mds,
            numMatched=len(mds),
            numReturned=len(mds),
            timestamp=gws.lib.datetimex.to_iso_string(with_tz=':'),
        )

        return self.template_response(
            sr,
            '',
            metadataCollection=mdc,
            next=0,
        )

    def handle_get_record_by_id(self, sr: server.request.Object):
        md = self._find_meta_by_id(sr)
        mds = [md] if md else []

        mdc = server.MetadataCollection(
            members=mds,
            numMatched=len(mds),
            numReturned=len(mds),
            timestamp=gws.lib.datetimex.to_iso_string(with_tz=':'),
        )

        return self.template_response(
            sr,
            '',
            metadataCollection=mdc,
            next=0,
        )

    ##

    def collect_metadata(self):
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
            self._collect_metadata_from_object(obj)

        gws.log.info(f'CSW: configured with {len(self.mdMap)} records')

    def _collect_metadata_from_object(self, obj: gws.Node):
        md: gws.Metadata = cast(gws.Metadata, gws.u.get(obj, 'metadata'))

        if not md:
            return

        if not md.get('catalogUid'):
            # gws.log.debug(f'CSW: skip {obj.uid}: no catalogUid')
            return

        cid = gws.u.to_uid(md.get('catalogUid'))

        if not self.root.app.authMgr.is_public_object(obj):
            gws.log.debug(f'CSW: skip {obj.uid}: not public')
            return

        extra = {}

        extra['catalogUid'] = cid
        extra['catalogCitationUid'] = md.get('catalogCitationUid') or cid
        extra['metaLinks'] = list(md.get('metaLinks') or [])
        extra['metaLinks'].append(self._make_link(cid))

        extent = gws.u.get(obj, 'extent') or gws.u.get(obj, 'map.extent')
        crs = gws.u.get(obj, 'crs') or gws.u.get(obj, 'map.crs')
        if extent and crs:
            extra['wgsExtent'] = gws.lib.extent.transform_to_wgs(extent, crs)
            extra['crs'] = crs
            # @TODO get boundingPolygonElement somehow

        map = obj.find_closest(gws.ext.object.map)
        if map:
            extra['isoSpatialResolution'] = gws.lib.uom.res_to_scale(cast(gws.base.map.Object, map).initResolution)

        self.mdMap[cid] = gws.base.metadata.from_args(md, extra)

    ##

    def _make_link(self, cid):
        return gws.MetadataLink(
            url=gws.u.action_url_path('owsService', serviceUid=self.uid, request='GetRecordById', id=cid),
            format=gws.lib.mime.XML,
            type='TC211' if self.profile == 'ISO' else 'DCMI',
            function='download',
        )

    def _find_metas(self, sr: server.request.Object):
        flt_el = None
        if sr.xmlElement:
            flt_el = sr.xmlElement.findfirst('Query/Constraint/Filter')

        if not flt_el:
            return self.mdMap.values()

        flt = gws.base.search.filter.from_fes_element(flt_el)
        m = gws.base.search.filter.Matcher()

        return [md for md in self.mdMap.values() if m.matches(flt, md)]

    def _find_meta_by_id(self, sr: server.request.Object):
        for md in self.mdMap.values():
            if md.catalogUid == sr.req.param('id'):
                return md
