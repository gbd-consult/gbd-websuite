"""WMS Service.

Implements WMS 1.1.x and 1.3.0.

Does not support SLD extensions except ``GetLegendGraphic``, for which only ``LAYERS`` is supported.
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
import gws.lib.bounds
import gws.lib.extent
import gws.lib.crs
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

_DEFAULT_MAX_PIXEL_SIZE = 2048


class Config(server.service.Config):
    """WMS Service configuration"""

    layerLimit: int = 0
    """WMS LayerLimit. (added in 8.1)"""
    maxPixelSize: int = 0
    """WMS MaxWidth/MaxHeight value. (added in 8.1)"""


class Object(server.service.Object):
    protocol = gws.OwsProtocol.WMS
    supportedVersions = ['1.3.0', '1.1.1', '1.1.0']
    isRasterService = True
    isOwsCommon = False

    layerLimit: int = 0
    maxPixelSize: int = 0

    def configure(self):
        self.layerLimit = self.cfg('layerLimit') or 0
        self.maxPixelSize = self.cfg('maxPixelSize') or _DEFAULT_MAX_PIXEL_SIZE

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.lib.metadata.merge(_DEFAULT_METADATA, self.metadata)

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(
                verb=gws.OwsVerb.GetCapabilities,
                formats=self.available_formats(gws.OwsVerb.GetCapabilities),
                handlerName='handle_get_capabilities',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetFeatureInfo,
                formats=self.available_formats(gws.OwsVerb.GetFeatureInfo),
                handlerName='handle_get_feature_info',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetMap,
                formats=self.available_formats(gws.OwsVerb.GetMap),
                handlerName='handle_get_map',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetLegendGraphic,
                formats=self.available_formats(gws.OwsVerb.GetLegendGraphic),
                handlerName='handle_get_legend_graphic',
            ),
        ]

    ##

    def init_request(self, req):
        sr = super().init_request(req)

        sr.crs = sr.requested_crs('CRS,SRS') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = sr.version < '1.3'

        return sr

    def layer_is_compatible(self, layer: gws.Layer):
        return layer.isGroup or layer.canRenderBox

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(
            sr,
            sr.requested_format('FORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_get_map(self, sr: server.request.Object):
        self.set_size_and_resolution(sr)

        lcs = self.requested_layer_caps(sr, 'LAYER,LAYERS', bottom_first=True)
        if not lcs:
            raise server.error.LayerNotDefined()

        mime = sr.requested_format('FORMAT')

        lcs = self.visible_layer_caps(sr, lcs)
        if not lcs:
            return self.image_response(sr, None, mime)

        s = sr.string_param('TRANSPARENT', values={'true', 'false'}, default='true')
        transparent = (s == 'true')

        gws.log.debug(f'get_map: layers={[lc.layer for lc in lcs]}')

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
            mapSize=(sr.pxSize[0], sr.pxSize[1], gws.Uom.px),
            planes=planes,
            project=self.project,
            user=sr.req.user,
        )

        mro = gws.gis.render.render_map(mri)

        return self.image_response(sr, mro.planes[0].image, mime)

    def handle_get_legend_graphic(self, sr: server.request.Object):
        # @TODO currently only support 'layer'
        lcs = self.requested_layer_caps(sr, 'LAYER,LAYERS', bottom_first=False)
        return self.render_legend(sr, lcs, sr.requested_format('FORMAT'))

    def handle_get_feature_info(self, sr: server.request.Object):
        self.set_size_and_resolution(sr)

        # @TODO top-first or bottom-first?
        lcs = self.requested_layer_caps(sr, 'QUERY_LAYERS', bottom_first=False)
        lcs = [lc for lc in lcs if lc.isSearchable]
        if not lcs:
            raise server.error.LayerNotQueryable()

        fc = self.get_features(sr, lcs)

        return self.template_response(
            sr,
            sr.requested_format('INFO_FORMAT'),
            featureCollection=fc,
        )

    ##

    def requested_layer_caps(self, sr: server.request.Object, param_name: str, bottom_first=False) -> list[server.LayerCaps]:
        # Order for GetMap is bottom-first (OGC 06-042 7.3.3.3):
        # A WMS shall render the requested layers by drawing the leftmost in the list bottommost, the next one over that, and so on.
        #
        # Our layers are always top-first. So, for each requested layer, if it is a leaf, we add it to a lcs, otherwise,
        # add group leaves in _reversed_ order. Finally, reverse the lcs list.

        lcs = []

        def add(name):
            for lc in sr.layerCapsList:
                if server.layer_caps.layer_name_matches(lc, name):
                    if lc.isGroup:
                        lcs.extend(reversed(lc.leaves) if bottom_first else lc.leaves)
                    else:
                        lcs.append(lc)
                    return True

        for name in sr.list_param(param_name):
            if not add(name):
                raise server.error.LayerNotDefined(name)

        if self.layerLimit and len(lcs) > self.layerLimit:
            raise server.error.InvalidParameterValue('LAYER')
        if not lcs:
            raise server.error.LayerNotDefined()

        return gws.u.uniq(reversed(lcs) if bottom_first else lcs)

    def get_features(self, sr: server.request.Object, lcs: list[server.LayerCaps]):

        lcs = self.visible_layer_caps(sr, lcs)
        if not lcs:
            return self.feature_collection(sr, lcs, 0, [])

        # @TODO validate and raise InvalidPoint

        x = sr.int_param('X,I')
        y = sr.int_param('Y,J')

        x = sr.bounds.extent[0] + (x * sr.resolution)
        y = sr.bounds.extent[3] - (y * sr.resolution)

        point = gws.base.shape.from_xy(x, y, sr.crs)

        search = gws.SearchQuery(
            project=sr.project,
            layers=[lc.layer for lc in lcs],
            limit=sr.requested_feature_count('FEATURE_COUNT'),
            resolution=sr.resolution,
            shape=point,
            tolerance=self.searchTolerance,
        )

        results = self.root.app.searchMgr.run_search(search, sr.req.user)
        return self.feature_collection(sr, lcs, len(results), results)

    def set_size_and_resolution(self, sr: server.request.Object):

        sr.bounds = sr.requested_bounds('BBOX')
        if not sr.bounds:
            raise server.error.MissingParameterValue('BBOX')

        sr.pxSize = sr.int_param('WIDTH'), sr.int_param('HEIGHT')
        if sr.pxSize[0] > self.maxPixelSize:
            raise server.error.InvalidParameterValue('WIDTH')
        if sr.pxSize[1] > self.maxPixelSize:
            raise server.error.InvalidParameterValue('HEIGHT')

        mw = gws.lib.bounds.width_in_meters(sr.bounds)

        dpi = sr.int_param('DPI', default=0) or sr.int_param('MAP_RESOLUTION', default=0)
        if dpi:
            # honor the dpi setting - compute the scale with "their" dpi and convert to "our" resolution
            sr.resolution = gws.lib.uom.scale_to_res(gws.lib.uom.mm_to_px(1000.0 * mw / sr.pxSize[0], dpi))
        else:
            sr.resolution = mw / sr.pxSize[0]

        gws.log.debug(f'set_size_and_resolution: {mw=} px={sr.pxSize} {dpi=} res={sr.resolution} 1:{gws.lib.uom.res_to_scale(sr.resolution)}')

    def visible_layer_caps(self, sr, lcs: list[server.LayerCaps]) -> list[server.LayerCaps]:
        return [
            lc for lc in lcs
            if min(lc.layer.resolutions) <= sr.resolution <= max(lc.layer.resolutions)
        ]
