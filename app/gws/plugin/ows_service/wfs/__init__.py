import gws
import gws.base.model
import gws.base.search.runner
import gws.base.web.error
import gws.lib.bounds
import gws.lib.extent
import gws.lib.gml
import gws.lib.mime
import gws.lib.os2
import gws.lib.ows.filter
import gws.lib.proj
import gws.lib.render
import gws.lib.shape
import gws.lib.xml2
import gws.types as t
from .. import core


@gws.ext.Config('ows.service.wfs')
class Config(core.ServiceConfig):
    """WFS Service configuration"""

    pass


@gws.ext.Object('ows.service.wfs')
class Object(core.Service):
    protocol = gws.OwsProtocol.WFS
    supported_versions = ['2.0.2', '2.0.1', '2.0.0']
    is_vector_ows = True

    @property
    def service_link(self):
        if self.project:
            return gws.Data(url=self.service_url_path(self.project), scheme='OGC:WFS', function='download')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/describeFeatureType.cx',
                subject='ows.DescribeFeatureType',
                mimeTypes=['xml'],
            ),
            gws.Config(
                type='xml',
                path=gws.dirname(__file__) + '/templates/getFeature.cx',
                subject='ows.GetFeatureInfo',
                mimeTypes=['xml', 'gml', 'gml3'],
            ),
        ]

    @property
    def default_metadata(self):
        return gws.Data(
            inspireDegreeOfConformity='notEvaluated',
            inspireMandatoryKeyword='infoMapAccessService',
            inspireResourceType='service',
            inspireSpatialDataServiceType='view',
            isoScope='dataset',
            isoSpatialRepresentationType='vector',
        )

    ##

    def configure(self):
        pass

    def handle_getcapabilities(self, rd: core.Request):
        lcs = self.layer_caps_list(rd)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')
        return self.template_response(rd, gws.OwsVerb.GetCapabilities, context={
            'layer_caps_list': lcs,
            'version': self.request_version(rd),
        })

    def handle_describefeaturetype(self, rd: core.Request):
        lcs = self.layer_caps_list_from_request(rd, ['typeName', 'typeNames'], self.SCOPE_FEATURE, fallback_to_all=True)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')

        # @TODO handle multiple namespaces

        return self.template_response(rd, gws.OwsVerb.DescribeFeatureType, context={
            'layer_caps_list': lcs,
            'namespace': lcs[0].feature_xname.ns,
            'version': self.request_version(rd),
        })

    def handle_getfeature(self, rd: core.Request):
        lcs = self.layer_caps_list_from_request(rd, ['typeName', 'typeNames'], self.SCOPE_FEATURE)
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

        flt: t.Optional[gws.SearchFilter] = None
        if rd.req.has_param('filter'):
            src = rd.req.param('filter')
            try:
                flt = gws.lib.ows.filter.from_fes_string(src)
            except gws.lib.ows.filter.Error as err:
                gws.log.error(f'FILTER ERROR: {err!r} filter={src!r}')
                raise gws.base.web.error.BadRequest('Invalid FILTER value')
            gws.p('FILTER', flt)

        result_type = rd.req.param('resultType', default='results').lower()
        if result_type not in ('hits', 'results'):
            raise gws.base.web.error.BadRequest('Invalid RESULTTYPE value')
        with_results = result_type == 'results'

        args = gws.SearchArgs(
            project=rd.project,
            shapes=[shape],
            filter=flt,
            layers=[lc.layer for lc in lcs],
            limit=limit,
            tolerance=(10, 'px'),
            resolution=1,
        )

        features = gws.base.search.runner.run(rd.req, args)

        coll = self.feature_collection(
            rd,
            features,
            lcs,
            populate=with_results,
            target_crs=request_crs,
            invert_axis_if_geographic=gws.lib.proj.invert_axis(crs_format),
            crs_format=crs_format)

        fmt = rd.req.param('output_format', default='gml')
        return self.template_response(rd, gws.OwsVerb.GetFeatureInfo, format=fmt, context={'collection': coll})
