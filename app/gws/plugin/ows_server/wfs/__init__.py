import gws
import gws.base.ows.server as server
import gws.base.search.runner
import gws.base.shape
import gws.base.web
import gws.gis.bounds
import gws.gis.crs
import gws.lib.metadata
import gws.lib.mime

gws.ext.new.owsService('wfs')

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='py',
        path=gws.dirname(__file__) + '/templates/getCapabilities.py',
        subject='ows.GetCapabilities',
        mimeTypes=['xml'],
        access=gws.PUBLIC,
    ),
    gws.Config(
        type='py',
        path=gws.dirname(__file__) + '/templates/describeFeatureType.py',
        subject='ows.DescribeFeatureType',
        mimeTypes=['xml'],
    ),
    gws.Config(
        type='py',
        path=gws.dirname(__file__) + '/templates/getFeature.py',
        subject='ows.GetFeatureInfo',
        mimeTypes=['xml', 'gml', 'gml3'],
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

    def configure_templates(self):
        super().configure_templates()
        self.templates.extend(self.configure_template(c) for c in _DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.lib.metadata.merge(_DEFAULT_METADATA, self.metadata)

    ##

    def handle_getcapabilities(self, rd: server.Request):
        return self.template_response(
            rd,
            gws.OwsVerb.GetCapabilities,
            rd.req.param('format') or gws.lib.mime.XML,
            layerCapsList=self.all_layer_caps(rd)
        )

    def handle_describefeaturetype(self, rd: server.Request):
        raise gws.base.web.error.NotFound('No layers found')
        # lcs = [
        #     lc for lc in self.requested_layer_caps(rd)
        #     if lc.model and lc.layer.owsOptions.xmlNamespace
        # ]
        # if not lcs:
        #     raise gws.base.web.error.NotFound('No layers found')
        #
        # return self.template_response(
        #     rd,
        #     gws.OwsVerb.DescribeFeatureType,
        #     rd.req.param('format') or gws.lib.mime.XML,
        #     layerCapsList=[lcs[0]]
        # )

    def handle_getfeature(self, rd: server.Request):
        lcs = self.requested_layer_caps(rd)

        try:
            limit = int(rd.req.param('count') or rd.req.param('maxFeatures') or 0)
        except Exception:
            raise gws.base.web.error.BadRequest('Invalid COUNT value')

        self.set_crs_and_bounds(rd)

        # flt: t.Optional[gws.SearchFilter] = None
        # if rd.req.has_param('filter'):
        #     src = rd.req.param('filter')
        #     try:
        #         flt = gws.gis.ows.filter.from_fes_string(src)
        #     except gws.gis.ows.filter.Error as err:
        #         gws.log.error(f'FILTER ERROR: {err!r} filter={src!r}')
        #         raise gws.base.web.error.BadRequest('Invalid FILTER value')

        search = gws.SearchQuery(
            project=rd.project,
            layers=[lc.layer for lc in lcs],
            limit=min(limit or self.searchMaxLimit, self.searchMaxLimit),
            shape=gws.base.shape.from_bounds(rd.bounds),
        )

        results = gws.base.search.runner.run(self.root, search, rd.req.user)

        result_type = rd.req.param('resultType', default='results').lower()
        if result_type not in ('hits', 'results'):
            raise gws.base.web.error.BadRequest('Invalid RESULTTYPE value')

        if result_type == 'results':
            fc = server.util.feature_collection(rd, results)
        else:
            fc = server.util.empty_feature_collection(rd, results)

        return self.template_response(
            rd,
            gws.OwsVerb.GetFeatureInfo,
            format=rd.req.param('info_format', default='gml'),
            featureCollection=fc)

    def all_layer_caps(self, rd: server.Request):
        lct = server.util.layer_caps_tree(rd, self.rootLayer)
        lcs = [lc for lc in lct.leaves if lc.layer.isSearchable]
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')
        return lcs

    def requested_layer_caps(self, rd: server.Request):
        lct = server.util.layer_caps_tree(rd, self.rootLayer)

        s = server.util.one_of_params(rd, 'typeName', 'typeNames')
        if not s:
            lcs = [lc for lc in lct.leaves if lc.layer.isSearchable]
            if not lcs:
                raise gws.base.web.error.NotFound('No layers found')
            return lcs

        lcs = server.util.layer_caps_by_feature_name(lct, gws.to_list(s))
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')
        return lcs

    def set_crs_and_bounds(self, rd: server.Request):
        rd.crs = self.requested_crs(rd) or rd.project.map.bounds.crs
        rd.targetCrs = rd.crs
        rd.alwaysXY = False
        if rd.req.param('bbox'):
            rd.bounds = self.requested_bounds(rd)
        else:
            rd.bounds = gws.gis.bounds.transform(rd.project.map.bounds, rd.crs)
