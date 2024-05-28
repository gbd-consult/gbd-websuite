from typing import Optional

import gws
import gws.base.ows.server as server
import gws.base.shape
import gws.base.web
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.lib.metadata
import gws.lib.mime

gws.ext.new.owsService('wfs')

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/templates/getCapabilities.cx.py',
        subject='ows.GetCapabilities',
        mimeTypes=[
            gws.lib.mime.XML,
        ],
    ),
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/templates/getFeature3.cx.py',
        subject='ows.GetFeature',
        mimeTypes=[
            gws.lib.mime.XML,
            gws.lib.mime.GML,
            gws.lib.mime.GML3,
        ],
    ),
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/templates/getFeature2.cx.py',
        subject='ows.GetFeature',
        mimeTypes=[
            gws.lib.mime.GML2,
        ],
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
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.lib.metadata.merge(_DEFAULT_METADATA, self.metadata)

    ##

    def activate(self):
        self.handlers = {
            gws.OwsVerb.GetCapabilities: self.handle_getcapabilities,
            gws.OwsVerb.DescribeFeatureType: self.handle_describefeaturetype,
            gws.OwsVerb.GetFeature: self.handle_getfeature,
        }

    ##

    def init_service_request(self, req):
        sr = super().init_service_request(req)

        sr.crs = self.requested_crs(sr, 'crsName', 'srsName') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = False
        if sr.req.param('bbox'):
            sr.bounds = self.requested_bounds(sr, 'bbox')
        else:
            sr.bounds = gws.gis.bounds.transform(sr.project.map.bounds, sr.crs)

        return sr

    ##

    def handle_getcapabilities(self, sr: server.ServiceRequest):
        return self.template_response(
            sr,
            format=sr.req.param('format') or gws.lib.mime.XML,
            layerCapsList=self.all_layer_caps(sr)
        )

    def handle_describefeaturetype(self, sr: server.ServiceRequest):
        lcs = [
            lc for lc in self.requested_layer_caps(sr)
            if lc.model and lc.xmlNamespace
        ]
        xml = server.util.xml_schema(lcs)
        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=xml.to_string(with_xml_declaration=True, with_namespace_declarations=True, with_schema_locations=True)
        )

    def handle_getfeature(self, sr: server.ServiceRequest):
        lcs = self.requested_layer_caps(sr)

        try:
            limit = int(sr.req.param('count') or sr.req.param('maxFeatures') or 0)
        except Exception:
            raise gws.base.web.error.BadRequest('Invalid COUNT value')

        # flt: Optional[gws.SearchFilter] = None
        # if sr.req.has_param('filter'):
        #     src = sr.req.param('filter')
        #     try:
        #         flt = gws.gis.ows.filter.from_fes_string(src)
        #     except gws.gis.ows.filter.Error as err:
        #         gws.log.error(f'FILTER ERROR: {err!r} filter={src!r}')
        #         raise gws.base.web.error.BadRequest('Invalid FILTER value')

        search = gws.SearchQuery(
            project=sr.project,
            layers=[lc.layer for lc in lcs],
            limit=min(limit or self.searchMaxLimit, self.searchMaxLimit),
            shape=gws.base.shape.from_bounds(sr.bounds),
        )

        results = self.root.app.searchMgr.run_search(search, sr.req.user)

        result_type = sr.req.param('resultType', default='results').lower()
        if result_type not in ('hits', 'results'):
            raise gws.base.web.error.BadRequest('Invalid RESULTTYPE value')

        if result_type == 'results':
            fc = server.util.feature_collection(sr, lcs, results)
        else:
            fc = server.util.empty_feature_collection(sr, results)

        return self.template_response(
            sr,
            format=sr.req.param('info_format'),
            featureCollection=fc)

    def all_layer_caps(self, sr: server.ServiceRequest):
        lct = server.util.layer_caps_tree(sr, self.rootLayer)
        lcs = [lc for lc in lct.leaves if lc.layer.isSearchable]
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')
        return lcs

    def requested_layer_caps(self, sr: server.ServiceRequest):
        lct = server.util.layer_caps_tree(sr, self.rootLayer)

        s = server.util.one_of_params(sr, 'typeName', 'typeNames')
        if not s:
            lcs = [lc for lc in lct.leaves if lc.layer.isSearchable]
            if not lcs:
                raise gws.base.web.error.NotFound('No layers found')
            return lcs

        lcs = server.util.layer_caps_by_feature_name(lct, gws.u.to_list(s))
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')
        return lcs

    # def set_crs_and_bounds(self, sr: server.ServiceRequest):
    #     sr.crs = self.requested_crs(sr) or sr.project.map.bounds.crs
    #     sr.targetCrs = sr.crs
    #     sr.alwaysXY = False
    #     if sr.req.param('bbox'):
    #         sr.bounds = self.requested_bounds(sr)
    #     else:
    #         sr.bounds = gws.gis.bounds.transform(sr.project.map.bounds, sr.crs)
