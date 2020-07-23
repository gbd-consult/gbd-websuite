import gws
import gws.common.model
import gws.common.ows.service as ows
import gws.common.search.runner
import gws.gis.extent
import gws.gis.filter
import gws.gis.gml
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.tools.os2
import gws.tools.xml2
import gws.tools.mime
import gws.web.error

import gws.types as t


class Config(ows.Config):
    """WFS Service configuration"""

    pass


class Object(ows.Base):

    @property
    def service_link(self):
        return t.MetaLink(url=self.url, scheme='OGC:WFS', function='download')

    @property
    def default_templates(self):
        return [
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/describeFeatureType.cx',
                subject='ows.DescribeFeatureType',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/getFeature.cx',
                subject='ows.GetFeatureInfo',
                mimeTypes=['xml', 'gml2'],
            ),
        ]

    @property
    def default_metadata(self):
        return t.Data(
            inspireDegreeOfConformity=t.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=t.MetaInspireMandatoryKeyword.infoMapAccessService,
            inspireResourceType=t.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=t.MetaInspireSpatialDataServiceType.view,
            isoScope=t.MetaIsoScope.dataset,
            isoSpatialRepresentationType=t.MetaIsoSpatialRepresentationType.vector,
        )

    @property
    def default_name(self):
        return 'WFS'

    ##

    def configure(self):
        super().configure()

        self.type = 'wfs'
        self.supported_versions = ['2.0.2', '2.0.1', '2.0.0']

    def handle_getcapabilities(self, rd: ows.Request):
        lcs = self._layer_caps_with_schemas(rd)
        return self.template_response(rd, 'GetCapabilities', context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_describefeaturetype(self, rd: ows.Request):
        lcs = self._layer_caps_with_schemas(rd)
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
            raise gws.web.error.NotFound('Invalid type name')
        try:
            limit = int(rd.req.param('count') or rd.req.param('maxFeatures') or 0)
        except:
            raise gws.web.error.BadRequest('Invalid COUNT value')

        crs = rd.req.param('srsName') or rd.project.map.crs

        if rd.req.has_param('bbox'):
            bbox = gws.gis.extent.from_string(rd.req.param('bbox'))
            if not bbox:
                raise gws.web.error.BadRequest('Invalid BBOX value')
            shape = gws.gis.shape.from_extent(extent=bbox, crs=crs)
        else:
            shape = gws.gis.shape.from_extent(extent=rd.project.map.extent, crs=rd.project.map.crs)

        if rd.req.has_param('filter'):
            src = rd.req.param('filter')
            try:
                filter = gws.gis.filter.from_fes_string(src)
            except gws.gis.filter.Error as err:
                gws.log.error(f'FILTER ERROR: {err!r} filter={src!r}')
                raise gws.web.error.BadRequest('Invalid FILTER value')
            gws.p('FILTER', filter)
        else:
            filter = None

        args = t.SearchArgs(
            project=rd.project,
            shapes=[shape],
            filter=filter,
            layers=[lc.layer for lc in lcs],
            limit=limit,
            tolerance=(10, 'px'),
            resolution=1,
        )

        features = gws.common.search.runner.run(rd.req, args)
        for f in features:
            f.transform_to(crs)

        fmt = rd.req.param('output_format') or gws.tools.mime.get('gml2')
        return self.template_response(rd, 'GetFeatureInfo', fmt, context={
            'collection': self.feature_collection(features, rd),
        })

    def _layer_caps_with_schemas(self, rd) -> t.List[ows.LayerCaps]:
        d = {}

        for lc in self.layer_caps_list(rd):
            if lc.feature_schema:
                d[lc.feature_name.q] = lc

        return list(d.values())
