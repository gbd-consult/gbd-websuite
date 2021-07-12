import gws
import gws.types as t
import gws.base.model
import gws.base.ows.service as ows
import gws.base.search.runner
import gws.lib.bounds
import gws.lib.extent
import gws.lib.filter
import gws.lib.gml
import gws.lib.proj
import gws.lib.render
import gws.lib.shape
import gws.lib.mime
import gws.lib.os2
import gws.lib.xml2
import gws.base.web.error


class Config(ows.Config):
    """WFS Service configuration"""

    pass


class Object(ows.Base):

    @property
    def service_link(self):
        if self.project:
            return gws.MetaLink(url=self.url_for_project(self.project), scheme='OGC:WFS', function='download')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/describeFeatureType.cx',
                subject='ows.DescribeFeatureType',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/getFeature.cx',
                subject='ows.GetFeatureInfo',
                mimeTypes=['xml', 'gml', 'gml3'],
            ),
        ]

    @property
    def default_metadata(self):
        return gws.Data(
            inspireDegreeOfConformity=gws.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=gws.MetaInspireMandatoryKeyword.infoMapAccessService,
            inspireResourceType=gws.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=gws.MetaInspireSpatialDataServiceType.view,
            isoScope=gws.MetaIsoScope.dataset,
            isoSpatialRepresentationType=gws.MetaIsoSpatialRepresentationType.vector,
        )

    @property
    def default_name(self):
        return 'WFS'

    ##

    def configure(self):
        

        self.type = 'wfs'
        self.supported_versions = ['2.0.2', '2.0.1', '2.0.0']

    def handle_getcapabilities(self, rd: ows.Request):
        lcs = self._filter_layer_caps(self.layer_caps_list(rd))
        return self.template_response(rd, 'GetCapabilities', context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_describefeaturetype(self, rd: ows.Request):
        lcs = self._filter_layer_caps(self.layer_caps_list_from_request(rd, ['typeName', 'typeNames']))
        if not lcs:
            raise gws.base.web.error.BadRequest('Invalid type name')

        # @TODO multiple namespaces should be handled by importing individual ns schemas
        return self.template_response(rd, 'DescribeFeatureType', context={
            'layer_caps_list': lcs,
            'ns': lcs[0].feature_name.ns if lcs else '',
            'ns_uri': lcs[0].feature_name.ns_uri if lcs else '',
            'version': self.request_version(rd),
        })

    def handle_getfeature(self, rd: ows.Request):
        lcs = self.layer_caps_list_from_request(rd, ['typeName', 'typeNames'])
        if not lcs:
            raise gws.base.web.error.BadRequest('Invalid type name')

        try:
            limit = int(rd.req.param('count') or rd.req.param('maxFeatures') or 0)
        except:
            raise gws.base.web.error.BadRequest('Invalid COUNT value')

        request_crs = rd.project.map.crs
        crs_format = 'uri'

        p = rd.req.param('srsName')
        if p:
            fmt, srid = gws.lib.proj.parse(p)
            if not srid:
                raise gws.base.web.error.BadRequest('Invalid CRS')
            request_crs = gws.lib.proj.format(srid, 'epsg')
            crs_format = fmt

        if rd.req.has_param('bbox'):
            bounds = gws.lib.bounds.from_request_bbox(rd.req.param('bbox'), request_crs, invert_axis_if_geographic=True)
            if not bounds:
                raise gws.base.web.error.BadRequest('Invalid BBOX')
            shape = gws.lib.shape.from_bounds(bounds)
            request_crs = shape.crs
        else:
            shape = gws.lib.shape.from_extent(extent=rd.project.map.extent, crs=rd.project.map.crs)

        if rd.req.has_param('filter'):
            src = rd.req.param('filter')
            try:
                filter = gws.lib.filter.from_fes_string(src)
            except gws.lib.filter.Error as err:
                gws.log.error(f'FILTER ERROR: {err!r} filter={src!r}')
                raise gws.base.web.error.BadRequest('Invalid FILTER value')
            gws.p('FILTER', filter)
        else:
            filter = None

        result_type = rd.req.param('resultType', default='results').lower()
        if result_type not in ('hits', 'results'):
            raise gws.base.web.error.BadRequest('Invalid RESULTTYPE value')
        with_results = result_type == 'results'

        args = gws.SearchArgs(
            project=rd.project,
            shapes=[shape],
            filter=filter,
            layers=[lc.layer for lc in lcs],
            limit=limit,
            tolerance=(10, 'px'),
            resolution=1,
        )

        features = gws.base.search.runner.run(rd.req, args)

        coll = self.feature_collection(
            features,
            rd,
            populate=with_results,
            target_crs=request_crs,
            invert_axis_if_geographic=gws.lib.proj.invert_axis(crs_format),
            crs_format=crs_format)

        return self.template_response(
            rd,
            'GetFeatureInfo',
            ows_format=rd.req.param('output_format') or 'gml',
            context={'collection': coll})

    def _filter_layer_caps(self, lcs) -> t.List[ows.LayerCaps]:
        # return only layer caps that have schemas
        d = {}
        for lc in lcs:
            if lc.feature_schema:
                d[lc.feature_name.q] = lc
        return list(d.values())
