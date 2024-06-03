"""WMS Service.

Implements WMS 1.1.x and 1.3.0.

Does not support SLD extensions except ``GetLegendGraphic``, for which only ``LAYERS`` is supported.

References:
    - OGC 01-068r3 (https://portal.ogc.org/files/?artifact_id=1081)
    - OGC 06-042 (https://portal.ogc.org/files/?artifact_id=14416)
    - https://mapserver.org/ogc/wms_server.html
    - https://docs.geoserver.org/latest/en/user/services/wms/reference.html
"""

# @TODO strict mode
#
# OGC 06-042 7.2.4.7.2
# A server shall issue a service exception (code="LayerNotQueryable") if GetFeatureInfo is requested on a Layer that is not queryable.

# OGC 06-042 7.2.4.6.3
# A server shall throw a service exception (code="LayerNotDefined") if an invalid layer is requested.

import gws
import gws.base.legend
import gws.base.ows.server as server
import gws.base.shape
import gws.base.web
import gws.config.util
import gws.gis.bounds
import gws.gis.extent
import gws.gis.crs
import gws.gis.render
import gws.lib.image
import gws.lib.metadata
import gws.lib.mime
import gws.lib.uom

gws.ext.new.owsService('wms')

_cdir = gws.u.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='py',
        path=f'{_cdir}/templates/getCapabilities.cx.py',
        subject='ows.GetCapabilities',
        mimeTypes=[gws.lib.mime.XML],
    ),
    # NB use the wfs template with GML2 (qgis doesn't understand GML3 for WMS)
    gws.Config(
        type='py',
        path=f'{_cdir}/../wfs/templates/getFeature2.cx.py',
        subject='ows.GetFeatureInfo',
        mimeTypes=[gws.lib.mime.GML2, gws.lib.mime.GML, gws.lib.mime.XML],
    )
]

_DEFAULT_METADATA = gws.Metadata(
    name='WMS',
    inspireDegreeOfConformity='notEvaluated',
    inspireMandatoryKeyword='infoMapAccessService',
    inspireResourceType='service',
    inspireSpatialDataServiceType='view',
    isoServiceFunction='search',
    isoScope='dataset',
    isoSpatialRepresentationType='vector',
)


class Config(server.service.Config):
    """WMS Service configuration"""

    layerLimit: int = 0
    """WMS LayerLimit value."""
    maxPixelSize: int = 0
    """WMS MaxWidth/MaxHeight value."""


class Object(server.service.Object):
    protocol = gws.OwsProtocol.WMS
    supportedVersions = ['1.3.0', '1.1.1', '1.1.0']
    isRasterService = True

    layerLimit: int = 0
    maxPixelSize: int = 0

    def configure(self):
        self.layerLimit = self.cfg('layerLimit') or 0
        self.maxPixelSize = self.cfg('layerLimit') or 0

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.lib.metadata.merge(_DEFAULT_METADATA, self.metadata)

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(verb=gws.OwsVerb.GetCapabilities, formats=self.template_formats(gws.OwsVerb.GetCapabilities)),
            gws.OwsOperation(verb=gws.OwsVerb.GetFeatureInfo, formats=self.template_formats(gws.OwsVerb.GetFeatureInfo)),
            gws.OwsOperation(verb=gws.OwsVerb.GetMap, formats=self.imageFormats),
            gws.OwsOperation(verb=gws.OwsVerb.GetLegendGraphic, formats=self.imageFormats),
        ]

    ##

    def activate(self):
        self.handlers = {
            gws.OwsVerb.GetCapabilities: self.handle_get_capabilities,
            gws.OwsVerb.GetFeatureInfo: self.handle_get_feature_info,
            gws.OwsVerb.GetLegendGraphic: self.handle_get_legend_graphic,
            gws.OwsVerb.GetMap: self.handle_get_map,
        }

    ##

    def init_request(self, req):
        sr = super().init_request(req)

        sr.crs = sr.get_crs('crs,srs') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = sr.version < '1.3'
        sr.bounds = sr.get_bounds('bbox')

        return sr

    def layer_is_suitable(self, layer: gws.Layer):
        return layer.isGroup or layer.canRenderBox

    def requested_layer_caps(self, sr: server.request.Object, param_name: str, bottom_first=True) -> list[server.LayerCaps]:
        # Order for GetMap is bottom-first:
        #
        # OGC 06-042 7.3.3.3
        # A WMS shall render the requested layers by drawing the leftmost in the list bottommost, the next one over that, and so on.
        #
        # @TODO: Assuming this holds for GetLegendGraphic and QUERY_LAYERS as well

        lcs = []

        for name in sr.list_param(param_name):
            for lc in sr.layerCapsList:
                if not server.layer_caps.layer_name_matches(lc, name):
                    continue
                if lc.isGroup:
                    lcs.extend(reversed(lc.leaves) if bottom_first else lc.leaves)
                else:
                    lcs.append(lc)

        if self.layerLimit and len(lcs) > self.layerLimit:
            raise gws.base.web.error.BadRequest('Too many layers')
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')

        return gws.u.uniq(reversed(lcs) if bottom_first else lcs)

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(
            sr,
            format=sr.string_param('format', default=''),
            layerCapsList=sr.layerCapsList,
        )

    def handle_get_map(self, sr: server.request.Object):
        return self.render_map(sr)

    def handle_get_legend_graphic(self, sr: server.request.Object):
        # @TODO currently only support 'layer'

        lcs = self.requested_layer_caps(sr, 'layer,layers')
        return sr.render_legend(lcs)

    def handle_get_feature_info(self, sr: server.request.Object):
        self.prepare_for_render(sr)

        lcs = self.requested_layer_caps(sr, 'query_layers')
        lcs = [lc for lc in lcs if lc.hasSearch]
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')

        fc = self.get_features(sr, lcs)

        return self.template_response(
            sr,
            format=sr.string_param('info_format', default=''),
            featureCollection=fc,
        )

    def get_features(self, sr, lcs):

        lcs = self.visible_layer_caps(sr, lcs)
        if not lcs:
            return sr.feature_collection(lcs, 0, [])

        x = sr.int_param('x,i')
        y = sr.int_param('y,j')

        x = sr.bounds.extent[0] + (x * sr.xResolution)
        y = sr.bounds.extent[3] - (y * sr.yResolution)

        point = gws.base.shape.from_xy(x, y, sr.crs)

        search = gws.SearchQuery(
            project=sr.project,
            layers=[lc.layer for lc in lcs],
            limit=sr.get_feature_count('feature_count'),
            resolution=sr.xResolution,
            shape=point,
            tolerance=self.searchTolerance,
        )

        results = self.root.app.searchMgr.run_search(search, sr.req.user)
        return sr.feature_collection(lcs, len(results), results)

    def render_map(self, sr: server.request.Object):
        self.prepare_for_render(sr)

        lcs = self.requested_layer_caps(sr, 'layer,layers')
        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')

        fmt = sr.string_param('format', default=self.imageFormats[0])
        if fmt and fmt not in self.imageFormats:
            raise gws.base.web.error.BadRequest('Invalid FORMAT')

        lcs = self.visible_layer_caps(sr, lcs)
        if not lcs:
            return gws.ContentResponse(
                mime=fmt,
                content=gws.lib.image.empty_pixel(fmt)
            )

        s = sr.string_param('transparent', values={'true', 'false'}, default='true')
        transparent = (s == 'true')

        planes = [
            gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.imageLayer,
                layer=lc.layer
            )
            for lc in lcs
        ]

        mri = gws.MapRenderInput(
            backgroundColor=None if transparent else 0,
            bbox=sr.bounds.extent,
            crs=sr.bounds.crs,
            mapSize=(sr.pxWidth, sr.pxHeight, gws.Uom.px),
            planes=planes,
            project=self.project,
            user=sr.req.user,
        )

        mro = gws.gis.render.render_map(mri)

        return gws.ContentResponse(
            mime=fmt,
            content=mro.planes[0].image.to_bytes(fmt, self.imageOptions)
        )

    def prepare_for_render(self, sr: server.request.Object):
        if not sr.bounds:
            raise gws.base.web.error.BadRequest('Missing BBOX parameter')

        sr.pxWidth = sr.int_param('width')
        sr.pxHeight = sr.int_param('height')
        w, h = gws.gis.extent.size(sr.bounds.extent)
        sr.xResolution = w / sr.pxWidth
        sr.yResolution = h / sr.pxHeight

        gws.log.debug(f'prepare_for_render: px={sr.pxWidth}x{sr.pxHeight} res={sr.xResolution} {gws.lib.uom.res_to_scale(sr.xResolution)}')

    def visible_layer_caps(self, sr, lcs: list[server.LayerCaps]) -> list[server.LayerCaps]:
        return [
            lc for lc in lcs
            if min(lc.layer.resolutions) <= sr.xResolution <= max(lc.layer.resolutions)
        ]
