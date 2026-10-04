"""WFS service.

Implements the WFS 2.0 "Basic" profile. Only ``GET`` requests with ``KVP``
encoding are supported.

The service publishes the project layers that are not groups, are searchable
and have an XML namespace. Supported operations are GetCapabilities,
DescribeFeatureType, GetFeature, GetPropertyValue, ListStoredQueries and
DescribeStoredQueries.

Supported ad hoc query parameters:

- ``TYPENAMES``
- ``SRSNAME``
- ``BBOX``
- ``STARTINDEX``
- ``COUNT``
- ``OUTPUTFORMAT``
- ``RESULTTYPE``

``FILTER`` and ``SORTBY`` are not supported yet.

Supported stored queries:

- ``urn:ogc:def:query:OGC-WFS::GetFeatureById``

For ``GetPropertyValue`` only a simple ``VALUEREFERENCE`` (a field name) is supported.

GetFeature runs a search with the ``BBOX`` of the request (or the map extent)
in the requested layers and applies ``STARTINDEX`` and ``COUNT`` to the results.
DescribeFeatureType returns an XML schema generated from the layers, unless
a template is configured. Features can be returned as GML 3, GML 2 or GeoJSON.

Each published layer needs an XML namespace, set with ``ows.featureName`` or
``ows.xmlns``; custom namespaces must also be configured globally (see
``gws.base.ows.server``).

Templates:

- ``templates/getCapabilities.cx.py``: ``ows.GetCapabilities``.
- ``templates/getFeature3.cx.py``: ``ows.GetFeature``, GML 3.
- ``templates/getFeatureGeoJson.cx.py``: ``ows.GetFeature``, GeoJSON.
- ``templates/getFeature2.cx.py``: ``ows.GetFeature``, GML 2.
- ``templates/getPropertyValue.cx.py``: ``ows.GetPropertyValue``.
- ``templates/listStoredQueries.cx.py``: ``ows.ListStoredQueries``.
- ``templates/describeStoredQueries.cx.py``: ``ows.DescribeStoredQueries``.

References:

- OGC 09-025r1 (https://portal.ogc.org/files/?artifact_id=39967)
- https://mapserver.org/ogc/wfs_server.html
- https://docs.geoserver.org/latest/en/user/services/wfs/reference.html

Example::

    projects+ {
        owsServices+ {
            type "wfs"
            uid "my_wfs"
        }
        map.layers+ {
            type "postgres"
            tableName "public.districts"
            ows.featureName "my:districts"
        }
    }
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


STORED_QUERY_GET_FEATURE_BY_ID = 'urn:ogc:def:query:OGC-WFS::GetFeatureById'
"""Identifier of the only supported stored query."""

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
    inspireMandatoryKeyword='infoFeatureAccessService',
    inspireResourceType='service',
    inspireSpatialDataServiceType='download',
    isoScope='dataset',
    isoServiceFunction='download',
    isoSpatialRepresentationType='vector',
)


@gws.ext.config.owsService('wfs')
class Config(server.service.Config):
    """WFS service that serves features of the project layers."""

    pass


@gws.ext.object.owsService('wfs')
class Object(server.service.Object):
    """WFS service that returns the features of the project layers."""

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
        sr.require_project()
        sr.crs = sr.requested_crs('CRSNAME,SRSNAME') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = False
        if sr.req.has_param('BBOX'):
            sr.bounds = gws.u.require(sr.requested_bounds('BBOX'))
        else:
            sr.bounds = gws.lib.bounds.transform(sr.project.map.bounds, sr.crs)

        return sr

    def layer_is_compatible(self, layer: gws.Layer):
        return not layer.isGroup and layer.isSearchable and layer.ows.xmlNamespace

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        """Handle the GetCapabilities operation.

        Args:
            sr: Service request.

        Returns:
            The capabilities document.
        """
        return self.template_response(
            sr,
            sr.requested_format('OUTPUTFORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_list_stored_queries(self, sr: server.request.Object):
        """Handle the ListStoredQueries operation.

        Args:
            sr: Service request.

        Returns:
            The list of stored queries.
        """
        return self.template_response(
            sr,
            sr.requested_format('FORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_describe_stored_queries(self, sr: server.request.Object):
        """Handle the DescribeStoredQueries operation.

        Args:
            sr: Service request.

        Returns:
            The description of the stored queries.

        Raises:
            ``server.error.InvalidParameterValue``: If ``STOREDQUERY_ID`` is not a supported query.
        """
        s = sr.string_param('STOREDQUERY_ID', default='')
        if s and s != STORED_QUERY_GET_FEATURE_BY_ID:
            raise server.error.InvalidParameterValue('STOREDQUERY_ID')

        return self.template_response(
            sr,
            sr.requested_format('OUTPUTFORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_describe_feature_type(self, sr: server.request.Object):
        """Handle the DescribeFeatureType operation.

        If a template is configured, it is used. Otherwise, an XML schema is
        generated for the requested layers.

        Args:
            sr: Service request.

        Returns:
            The XML schema of the requested feature types.
        """
        tpl = self.get_template(sr)
        if tpl:
            return self.template_response(sr)

        # if no template is defined, we return the XML schema for the requested layers

        lcs = self.requested_layer_caps(sr)
        el, opts = server.layer_caps.xml_schema(lcs, sr.req.user)

        opts.withNamespaceDeclarations = True
        opts.withSchemaLocations = True
        opts.withXmlDeclaration = True

        return self.xml_response(el, opts)

    def handle_get_feature(self, sr: server.request.Object):
        """Handle the GetFeature operation.

        Args:
            sr: Service request.

        Returns:
            The feature collection in the requested ``OUTPUTFORMAT``.
        """
        fc = self.get_features(sr)
        return self.template_response(sr, sr.requested_format('OUTPUTFORMAT'), featureCollection=fc)

    def handle_get_property_value(self, sr: server.request.Object):
        """Handle the GetPropertyValue operation.

        Args:
            sr: Service request.

        Returns:
            The values of the ``VALUEREFERENCE`` attribute of the found features.
        """
        value_ref = sr.string_param('VALUEREFERENCE')
        fc = self.get_features(sr)
        fc.values = [m.feature.get(value_ref) for m in fc.members]
        return self.template_response(sr, sr.requested_format('OUTPUTFORMAT'), featureCollection=fc)

    ##

    def requested_layer_caps(self, sr: server.request.Object):
        """Find the layer caps for the feature types in ``TYPENAME`` or ``TYPENAMES``.

        Args:
            sr: Service request.

        Returns:
            The matching layer caps without duplicates, or all layer caps if no feature types are requested.

        Raises:
            ``server.error.LayerNotDefined``: If none of the requested feature types is found.
        """
        tns = sr.list_param('TYPENAME,TYPENAMES')
        if not tns:
            return sr.layerCapsList
        lcs = []
        for name in tns:
            for lc in sr.layerCapsList:
                if server.layer_caps.feature_name_matches(lc, name, sr.customNamespacePrefixes):
                    lcs.append(lc)
        if not lcs:
            raise server.error.LayerNotDefined()
        return gws.u.uniq(lcs)

    SEARCH_MAX_TOTAL = 100_000
    """Max. number of features a search can return, before paging."""

    def get_features(self, sr: server.request.Object, value_ref: str = '') -> server.FeatureCollection:
        """Search for the features requested by GetFeature or GetPropertyValue.

        With ``RESULTTYPE=hits``, only the number of found features is returned.
        Otherwise, ``STARTINDEX`` and ``COUNT`` (or ``MAXFEATURES``) are applied
        to the results.

        Args:
            sr: Service request.
            value_ref: If given, only features that have this attribute are returned.

        Returns:
            The feature collection.
        """
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
        """Create the search query for a request.

        For the ``GetFeatureById`` stored query, the search looks up the feature
        given by the ``id`` parameter. Otherwise, it searches within the bounds
        of the request.

        Args:
            sr: Service request.
            lcs: Layer caps to search.

        Returns:
            The search query.

        Raises:
            ``server.error.InvalidParameterValue``: If ``STOREDQUERY_ID`` is not a supported query.
        """
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
