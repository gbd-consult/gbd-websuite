import gws
import gws.common.metadata
import gws.common.search.runner
import gws.gis.bounds
import gws.gis.extent
import gws.gis.gml
import gws.gis.legend
import gws.gis.proj
import gws.gis.shape
import gws.tools.date
import gws.tools.misc
import gws.tools.mime
import gws.tools.os2
import gws.tools.xml2
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows


class Config(gws.common.ows.service.Config):
    """WMS Service configuration"""
    pass


_GETMAP_MANDATORY_PARAMS = ['version', 'request', 'layers', 'styles', 'bbox', 'width', 'height', 'format']


class Object(ows.Base):

    @property
    def service_link(self):
        if self.project:
            return t.MetaLink(url=self.url_for_project(self.project), scheme='OGC:WMS', function='search')

    @property
    def default_templates(self):
        return [
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wms/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
            t.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wfs/templates/getFeature.cx',  # NB use the wfs template
                subject='ows.GetFeatureInfo',
                mimeTypes=['xml', 'gml', 'gml3'],
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
        return 'WMS'

    ##

    def configure(self):
        super().configure()

        self.type = 'wms'
        self.supported_versions = ['1.3.0', '1.1.1', '1.1.0']
        self.search_max_limit = 100

    ##

    def handle_getcapabilities(self, rd: ows.Request):
        # OGC 06-042, 7.2.3.5
        update_sequence = rd.req.param('updatesequence')
        if update_sequence and self.update_sequence and update_sequence >= self.update_sequence:
            raise gws.web.error.BadRequest('Wrong update sequence')

        root = self.layer_root_caps(rd)
        if not root:
            gws.log.debug(f'service={self.uid!r}: no layer_root_caps')
            raise gws.web.error.NotFound()

        fmt = rd.req.param('format') or gws.tools.mime.get('xml')

        supported_formats = self.enum_template_formats()
        supported_formats['getmap'] = ['image/png']
        supported_formats['getlegendgraphic'] = ['image/png']

        return self.template_response(rd, 'GetCapabilities', fmt, context={
            'layer_root_caps': root,
            'supported_formats': supported_formats,
            'version': self.request_version(rd),
        })

    def handle_getmap(self, rd: ows.Request):
        if self.strict_params:
            for p in _GETMAP_MANDATORY_PARAMS:
                if not rd.req.has_param(p):
                    raise gws.web.error.BadRequest(f'Required parameter missing: {p}')
            if not rd.req.has_param('crs') and not rd.req.has_param('srs'):
                raise gws.web.error.BadRequest('Required parameter missing: CRS')

        trans = rd.req.param('transparent', '').lower()
        if trans:
            if self.strict_params and trans not in ('true', 'false'):
                raise gws.web.error.BadRequest('Invalid parameter: TRANSPARENT')

        bounds = self._request_bounds(rd)
        if not bounds:
            raise gws.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'])
        if not lcs:
            raise gws.web.error.NotFound('No layers found')

        return self.render_map_bbox_from_layer_caps_list(lcs, bounds, rd)

    def handle_getlegendgraphic(self, rd: ows.Request):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'])
        if not lcs:
            raise gws.web.error.NotFound('No layers found')
        paths = [lc.layer.render_legend() for lc in lcs if lc.has_legend]
        out = gws.gis.legend.combine_legend_paths(paths)
        return t.HttpResponse(mime='image/png', content=out or gws.tools.misc.Pixels.png8)

    def handle_getfeatureinfo(self, rd: ows.Request):
        bounds = self._request_bounds(rd)
        if not bounds:
            raise gws.web.error.BadRequest('Invalid BBOX')

        lcs = self.layer_caps_list_from_request(rd, ['query_layers'])
        if not lcs:
            raise gws.web.error.NotFound('No layers found')

        try:
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
            limit = int(rd.req.param('feature_count', '1'))
            x = int(rd.req.param('i') or rd.req.param('x'))
            y = int(rd.req.param('j') or rd.req.param('y'))
        except:
            raise gws.web.error.BadRequest('Invalid parameter')

        bbox = bounds.extent
        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (x * xres)
        y = bbox[3] - (y * yres)

        point = gws.gis.shape.from_geometry({
            'type': 'Point',
            'coordinates': [x, y]
        }, bounds.crs)

        # @TODO: should be a parameter
        pixel_tolerance = 10

        args = t.SearchArgs(
            project=rd.project,
            layers=[lc.layer for lc in lcs],
            limit=min(limit, self.search_max_limit),
            resolution=xres,
            shapes=[point],
            tolerance=(pixel_tolerance, 'px'),
        )

        features = gws.common.search.runner.run(rd.req, args)

        coll = self.feature_collection(
            features,
            rd,
            target_crs=bounds.crs,
            invert_axis_if_geographic=self.request_version(rd) >= '1.3.0',
            crs_format='urn',
        )

        return self.template_response(
            rd,
            'GetFeatureInfo',
            ows_format=rd.req.param('info_format') or 'gml',
            context={'collection': coll})

    ###

    def _request_bounds(self, rd: ows.Request):
        ver = self.request_version(rd)
        return gws.gis.bounds.from_request_bbox(
            rd.req.param('bbox'),
            rd.req.param('crs') or rd.req.param('srs') or rd.project.map.crs,
            invert_axis_if_geographic=ver >= '1.3.0')
