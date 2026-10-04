"""CSW service.

Basic implementation of the OGC Catalogue Service for the Web (CSW) 2.0.2.
Only a small subset of the standard is supported: the operations
GetCapabilities, DescribeRecord, GetRecords and GetRecordById, with records
in the ISO 19115 profile. The ``profile`` option is not evaluated; the profile
is always ISO.

The catalog is collected after the configuration: every object in the tree
that has metadata with a ``catalogUid`` and is public becomes a record. A link
to the GetRecordById URL of the record is added to its ``metaLinks``. If the
object or its map has an extent and a CRS, the WGS84 extent is added, and the
initial scale of the closest map becomes the spatial resolution.

GetRecords supports an FES filter in ``Query/Constraint/Filter`` of an XML
request; without a filter all records are returned. GetRecordById looks up
a record by the ``id`` parameter.

Templates:

- ``templates/iso/getCapabilities.cx.py``: ``ows.GetCapabilities``.
- ``templates/iso/describeRecord.cx.py``: ``ows.DescribeRecord``.
- ``templates/iso/getRecords.cx.py``: ``ows.GetRecords`` and ``ows.GetRecordById``.

References:

- OpenGIS Catalogue Service Implementation Specification 2.0.2 (http://portal.opengeospatial.org/files/?artifact_id=20555)

Example::

    owsServices+ {
        type "csw"
        uid "my_csw"
        metadata {
            title "Catalog"
            abstract "Catalog of the geodata"
        }
    }

    projects+ {
        access "allow all"
        metadata.catalogUid "my_project"
    }
"""

from typing import cast
import gws
import gws.base.metadata
import gws.base.map
import gws.base.ows.server as server
import gws.base.search.filter
import gws.lib.shape
import gws.config.util
import gws.lib.crs
import gws.lib.datetimex
import gws.lib.extent
import gws.lib.mime
import gws.gis.zoom


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
    """Metadata profile for the CSW service."""

    ISO = 'ISO'
    """ISO 19115 metadata profile."""
    DCMI = 'DCMI'
    """Dublin Core metadata profile."""


@gws.ext.config.owsService('csw')
class Config(server.service.Config):
    """Catalogue service that publishes metadata records."""

    profile: Profile = Profile.ISO
    """Metadata profile."""


@gws.ext.object.owsService('csw')
class Object(server.service.Object):
    """CSW service that publishes the metadata of public objects as catalog records."""

    protocol = gws.OwsProtocol.CSW
    supportedVersions = ['2.0.2']

    mdMap: dict[str, gws.Metadata]
    """Catalog records by catalog UID."""
    profile: Profile
    """Metadata profile of the records."""

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
        """Handle the GetCapabilities operation.

        Args:
            sr: Service request.

        Returns:
            The capabilities document.
        """
        return self.template_response(sr)

    def handle_describe_record(self, sr: server.request.Object):
        """Handle the DescribeRecord operation.

        Args:
            sr: Service request.

        Returns:
            The record schema description.
        """
        return self.template_response(sr)

    def handle_get_records(self, sr: server.request.Object):
        """Handle the GetRecords operation.

        Returns all records that match the filter of the request.

        Args:
            sr: Service request.

        Returns:
            The search results document.
        """
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
        """Handle the GetRecordById operation.

        Args:
            sr: Service request.

        Returns:
            A response with the record given by the ``id`` parameter, or with no records.
        """
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
        """Collect the catalog records from all objects in the tree into ``mdMap``."""
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
        """Add a catalog record for an object, if it has a catalog UID and is public."""
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
            m = cast(gws.base.map.Object, map)
            extra['isoSpatialResolution'] = gws.gis.zoom.res_to_scale(m.initResolution, m.bounds.crs)

        self.mdMap[cid] = gws.base.metadata.from_args(md, extra)

    ##

    def _make_link(self, cid):
        """Create a metadata link to the GetRecordById URL of a record."""
        return gws.MetadataLink(
            url=gws.u.action_url_path('owsService', serviceUid=self.uid, request='GetRecordById', id=cid),
            format=gws.lib.mime.XML,
            type='TC211' if self.profile == 'ISO' else 'DCMI',
            function='download',
        )

    def _find_metas(self, sr: server.request.Object):
        """Return the records that match the filter of an XML request, or all records."""
        flt_el = None
        if sr.xmlElement:
            flt_el = sr.xmlElement.findfirst('Query/Constraint/Filter')

        if not flt_el:
            return self.mdMap.values()

        flt = gws.base.search.filter.from_fes_element(flt_el)
        m = gws.base.search.filter.Matcher()

        return [md for md in self.mdMap.values() if m.matches(flt, md)]

    def _find_meta_by_id(self, sr: server.request.Object):
        """Return the record with the catalog UID given by the ``id`` parameter."""
        for md in self.mdMap.values():
            if md.catalogUid == sr.req.param('id'):
                return md
