"""WFS Service.

Implements WFS 2.0 "Basic" profile.
This implementation only supports ``GET`` requests with ``KVP`` encoding.

Supported ad hoc query parameters:

- ``TYPENAMES``
- ``SRSNAME``
- ``BBOX``
- ``STARTINDEX``
- ``COUNT``
- ``OUTPUTFORMAT``
- ``RESULTTYPE``

@TODO: FILTER, SORTBY

Supported stored queries:

- ``urn:ogc:def:query:OGC-WFS::GetFeatureById``

For ``GetPropertyValue`` only simple ``VALUEREFERENCE`` (field name) is supported.

References:
    - OGC 09-025r1 (https://portal.ogc.org/files/?artifact_id=39967)
    - https://mapserver.org/ogc/wfs_server.html
    - https://docs.geoserver.org/latest/en/user/services/wfs/reference.html

"""

import gws
import gws.base.ows.server as server
import gws.base.shape
import gws.base.web
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.base.metadata
import gws.lib.mime

gws.ext.new.owsService('wfs')

STORED_QUERY_GET_FEATURE_BY_ID = "urn:ogc:def:query:OGC-WFS::GetFeatureById"

_cdir = gws.u.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/getCapabilities.cx.py',
        subject='ows.GetCapabilities',
        mimeTypes=[gws.lib.mime.XML],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/getFeature3.cx.py',
        subject='ows.GetFeature',
        access=gws.c.PUBLIC,
        mimeTypes=[gws.lib.mime.XML, gws.lib.mime.GML, gws.lib.mime.GML3],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/getFeatureGeoJson.cx.py',
        subject='ows.GetFeature',
        access=gws.c.PUBLIC,
        mimeTypes=[gws.lib.mime.JSON, gws.lib.mime.GEOJSON],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/getFeature2.cx.py',
        subject='ows.GetFeature',
        mimeTypes=[gws.lib.mime.GML2],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/getPropertyValue.cx.py',
        subject='ows.GetPropertyValue',
        mimeTypes=[gws.lib.mime.XML, gws.lib.mime.GML, gws.lib.mime.GML3],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/listStoredQueries.cx.py',
        subject='ows.ListStoredQueries',
        mimeTypes=[gws.lib.mime.XML],
    ),
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/describeStoredQueries.cx.py',
        subject='ows.DescribeStoredQueries',
        mimeTypes=[gws.lib.mime.XML],
    ),
]

_DEFAULT_METADATA = gws.Metadata(
    name='WFS',
    inspireMandatoryKeyword='infoMapAccessService',
    inspireResourceType='service',
    inspireSpatialDataServiceType='view',
    isoScope='dataset',
    isoServiceFunction='download',
    isoSpatialRepresentationType='vector',
)


class Config(server.service.Config):
    """WFS Service configuration"""
    pass


class Object(server.service.Object):
    protocol = gws.OwsProtocol.WFS
    supportedVersions = ['2.0.2', '2.0.1', '2.0.0']
    isVectorService = True
    isOwsCommon = True

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.base.metadata.from_args(_DEFAULT_METADATA, self.metadata)

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(
                verb=gws.OwsVerb.DescribeFeatureType,
                formats=[gws.lib.mime.GML3],
                handlerName='handle_describe_feature_type',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.DescribeStoredQueries,
                formats=self.available_formats(gws.OwsVerb.DescribeStoredQueries),
                handlerName='handle_describe_stored_queries',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetCapabilities,
                formats=self.available_formats(gws.OwsVerb.GetCapabilities),
                handlerName='handle_get_capabilities',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetFeature,
                formats=self.available_formats(gws.OwsVerb.GetFeature),
                handlerName='handle_get_feature',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetPropertyValue,
                formats=self.available_formats(gws.OwsVerb.GetPropertyValue),
                handlerName='handle_get_property_value',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.ListStoredQueries,
                formats=self.available_formats(gws.OwsVerb.ListStoredQueries),
                handlerName='handle_list_stored_queries',
            ),
        ]

    ##

    def init_request(self, req):
        sr = super().init_request(req)
        sr.load_project()

        sr.crs = sr.requested_crs('CRSNAME,SRSNAME') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = False
        sr.bounds = (
            sr.requested_bounds('BBOX') if sr.req.has_param('BBOX')
            else gws.lib.bounds.transform(sr.project.map.bounds, sr.crs)
        )

        return sr

    def layer_is_compatible(self, layer: gws.Layer):
        return not layer.isGroup and layer.isSearchable and layer.ows.xmlNamespace

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(
            sr,
            sr.requested_format('OUTPUTFORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_list_stored_queries(self, sr: server.request.Object):
        return self.template_response(
            sr,
            sr.requested_format('FORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_describe_stored_queries(self, sr: server.request.Object):
        s = sr.string_param('STOREDQUERY_ID', default='')
        if s and s != STORED_QUERY_GET_FEATURE_BY_ID:
            raise server.error.InvalidParameterValue('STOREDQUERY_ID')

        return self.template_response(
            sr,
            sr.requested_format('OUTPUTFORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_describe_feature_type(self, sr: server.request.Object):
        lcs = self.requested_layer_caps(sr)
        xml = server.layer_caps.xml_schema(lcs, sr.req.user)
        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=xml.to_string(
                with_xml_declaration=True,
                with_namespace_declarations=True,
                with_schema_locations=True,
            )
        )

    def handle_get_feature(self, sr: server.request.Object):
        fc = self.get_features(sr)
        return self.template_response(
            sr,
            sr.requested_format('OUTPUTFORMAT'),
            featureCollection=fc
        )

    def handle_get_property_value(self, sr: server.request.Object):
        value_ref = sr.string_param('VALUEREFERENCE')
        fc = self.get_features(sr)
        fc.values = [m.feature.get(value_ref) for m in fc.members]
        return self.template_response(
            sr,
            sr.requested_format('OUTPUTFORMAT'),
            featureCollection=fc
        )

    ##

    def requested_layer_caps(self, sr: server.request.Object):
        lcs = []
        for name in sr.list_param('TYPENAME,TYPENAMES'):
            for lc in sr.layerCapsList:
                if server.layer_caps.feature_name_matches(lc, name, sr.xmlnsReplacements):
                    lcs.append(lc)
        if not lcs:
            raise server.error.LayerNotDefined()
        return gws.u.uniq(lcs)

    SEARCH_MAX_TOTAL = 100_000

    def get_features(self, sr: server.request.Object, value_ref: str = '') -> server.FeatureCollection:

        # @TODO optimize paging for db-based layers

        lcs = self.requested_layer_caps(sr)
        search = self.make_search(sr, lcs)

        results = self.root.app.searchMgr.run_search(search, sr.req.user)

        if value_ref:
            results = [r for r in results if r.feature.has(value_ref)]

        hits = len(results)

        result_type = sr.string_param('RESULTTYPE', values={'hits', 'results'}, default='results')
        if result_type == 'hits':
            return self.feature_collection(sr, lcs, hits, [])

        limit = sr.requested_feature_count('COUNT,MAXFEATURES')
        offset = sr.int_param('STARTINDEX', default=0)

        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return self.feature_collection(sr, lcs, hits, results)

    def make_search(self, sr: server.request.Object, lcs):
        search = gws.SearchQuery(
            project=sr.project,
            layers=[lc.layer for lc in lcs],
            limit=self.SEARCH_MAX_TOTAL,
        )

        s = sr.string_param('STOREDQUERY_ID', default='')
        if s:
            if s != STORED_QUERY_GET_FEATURE_BY_ID:
                raise server.error.InvalidParameterValue('STOREDQUERY_ID')
            uid = sr.string_param('id')
            search.uids = [uid]
            return search

        # @TODO filters
        # flt: Optional[gws.SearchFilter] = None
        # if sr.req.has_param('filter'):
        #     src = sr.req.param('filter')
        #     try:
        #         flt = gws.gis.ows.filter.from_fes_string(src)
        #     except gws.gis.ows.filter.Error as err:
        #         gws.log.error(f'FILTER ERROR: {err!r} filter={src!r}')
        #         raise gws.base.web.error.BadRequest('Invalid FILTER value')

        search.shape = gws.base.shape.from_bounds(sr.bounds)
        return search
