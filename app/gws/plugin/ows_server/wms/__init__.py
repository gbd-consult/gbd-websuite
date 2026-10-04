"""WMS service.

Implements WMS 1.1.x and 1.3.0 with the operations GetCapabilities, GetMap,
GetFeatureInfo and GetLegendGraphic. SLD extensions are not supported, except
``GetLegendGraphic``, for which only the ``LAYER``/``LAYERS`` parameter is evaluated.

The service publishes the project layers that are groups or can render a box.
A requested group stands for its leaf layers. GetMap renders the requested
layers that are visible at the requested resolution into one image.
GetFeatureInfo runs a search at the clicked point in the queryable layers and
returns the results in GML 2, using the GetFeature template of the WFS service.
``DPI`` or ``MAP_RESOLUTION`` parameters are taken into account when the
request resolution is computed.

Templates:

- ``templates/getCapabilities.cx.py``: ``ows.GetCapabilities``.
- ``../wfs/templates/getFeature2.cx.py``: ``ows.GetFeatureInfo``.

Example::

    projects+ {
        owsServices+ {
            type "wms"
            uid "my_wms"
            supportedCrs [3857 25832]
            layerLimit 20
            maxPixelSize 4096
        }
    }
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
import gws.lib.shape
import gws.base.web
import gws.config.util
import gws.lib.bounds
import gws.lib.extent
import gws.lib.crs
import gws.gis.render
import gws.lib.image
import gws.base.metadata
import gws.lib.mime
import gws.lib.uom
import gws.gis.zoom


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
    ),
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


@gws.ext.config.owsService('wms')
class Config(server.service.Config):
    """WMS service that renders the project layers."""

    layerLimit: int = 0
    """Max. number of layers in a map request."""
    maxPixelSize: int = 0
    """Max. width and height of a map image in pixels."""


@gws.ext.object.owsService('wms')
class Object(server.service.Object):
    """WMS service that renders the project layers as map images and returns feature info and legends."""

    protocol = gws.OwsProtocol.WMS
    supportedVersions = ['1.3.0', '1.1.1', '1.1.0']
    isRasterService = True
    isOwsCommon = False

    layerLimit: int = 0
    """Max. number of layers in a request, 0 for no limit."""
    maxPixelSize: int = 0
    """Max. width and height of a map image in pixels, 2048 by default."""

    def configure(self):
        self.layerLimit = self.cfg('layerLimit') or 0
        self.maxPixelSize = self.cfg('maxPixelSize') or _DEFAULT_MAX_PIXEL_SIZE

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def configure_metadata(self):
        super().configure_metadata()
        self.metadata = gws.base.metadata.from_args(_DEFAULT_METADATA, self.metadata)

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(
                verb=gws.OwsVerb.GetCapabilities,
                formats=self.available_formats(gws.OwsVerb.GetCapabilities),
                handlerName='handle_get_capabilities',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetMap,
                formats=self.available_formats(gws.OwsVerb.GetMap),
                handlerName='handle_get_map',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetFeatureInfo,
                formats=self.available_formats(gws.OwsVerb.GetFeatureInfo),
                handlerName='handle_get_feature_info',
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
        sr.require_project()
        sr.crs = sr.requested_crs('CRS,SRS') or sr.project.map.bounds.crs
        sr.targetCrs = sr.crs
        sr.alwaysXY = sr.version < '1.3'
        return sr

    def layer_is_compatible(self, layer: gws.Layer):
        return layer.isGroup or layer.canRenderBox

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
            sr.requested_format('FORMAT'),
            layerCapsList=sr.layerCapsList,
        )

    def handle_get_map(self, sr: server.request.Object):
        """Handle the GetMap operation.

        Layers that are not visible at the requested resolution are skipped;
        if none remain, an empty image is returned. The background is
        transparent unless ``TRANSPARENT=false`` is requested.

        Args:
            sr: Service request.

        Returns:
            The map image.

        Raises:
            ``server.error.LayerNotDefined``: If no layers are requested.
        """
        self.set_size_and_resolution(sr)

        lcs = self.requested_layer_caps(sr, 'LAYER,LAYERS', bottom_first=True)
        if not lcs:
            raise server.error.LayerNotDefined()

        mime_type = sr.requested_format('FORMAT')

        lcs = self.visible_layer_caps(sr, lcs)
        if not lcs:
            return self.image_response(sr, None, mime_type)

        s = sr.string_param('TRANSPARENT', values={'true', 'false'}, default='true')
        transparent = s == 'true'

        gws.log.debug(f'get_map: layers={[lc.layer for lc in lcs]}')

        planes = [
            gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.imageLayer,
                layer=lc.layer,
            )
            for lc in lcs
        ]

        mri = gws.MapRenderInput(
            backgroundColor=None if transparent else 0,
            bbox=sr.bounds.extent,
            targetCrs=sr.bounds.crs,
            mapSize=(sr.pxSize[0], sr.pxSize[1], gws.Uom.px),
            planes=planes,
            project=self.project,
            user=sr.req.user,
        )

        mro = gws.gis.render.render_map(mri)

        return self.image_response(sr, mro.planes[0].image, mime_type)

    def handle_get_legend_graphic(self, sr: server.request.Object):
        """Handle the GetLegendGraphic operation.

        Args:
            sr: Service request.

        Returns:
            The legend image of the requested layers.
        """
        # @TODO currently only support 'layer'
        lcs = self.requested_layer_caps(sr, 'LAYER,LAYERS', bottom_first=False)
        return self.render_legend(sr, lcs, sr.requested_format('FORMAT'))

    def handle_get_feature_info(self, sr: server.request.Object):
        """Handle the GetFeatureInfo operation.

        Args:
            sr: Service request.

        Returns:
            The feature collection, rendered in the requested ``INFO_FORMAT``.

        Raises:
            ``server.error.LayerNotQueryable``: If none of the requested layers is searchable.
        """
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
        """Find the layer caps for the layer names in a request parameter.

        A requested group is replaced by its leaf layers. Duplicates are removed.

        Args:
            sr: Service request.
            param_name: Request parameter with the layer names, e.g. ``LAYERS``.
            bottom_first: Return the layers bottom-first, as required for GetMap.
                Otherwise, the layers are returned top-first.

        Returns:
            The layer caps.

        Raises:
            ``server.error.LayerNotDefined``: If a layer name is unknown or no layers are requested.
            ``server.error.InvalidParameterValue``: If more layers than ``layerLimit`` are requested.
        """
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
        """Search for features at the point given by the ``I``/``J`` (or ``X``/``Y``) parameters.

        Only layers visible at the requested resolution are searched. The search
        uses the search tolerance of the service.

        Args:
            sr: Service request, with bounds and resolution already set.
            lcs: Layer caps to search.

        Returns:
            The feature collection.
        """
        lcs = self.visible_layer_caps(sr, lcs)
        if not lcs:
            return self.feature_collection(sr, lcs, 0, [])

        # OGC 06-042, 7.4.3.7
        # - the point I=0, J=0 indicates the pixel at the upper left corner of the map;
        # - I increases to the right and J increases downward.
        # similar OGC 01-068r3, 7.3.3.8

        # @TODO validate and raise InvalidPoint

        ox = sr.int_param('X,I')
        oy = sr.int_param('Y,J')

        dx = ox * sr.resX
        dy = oy * sr.resY

        xy = sr.bounds.extent[0], sr.bounds.extent[3]
        xy = sr.bounds.crs.point_offset_in_meters(xy, dx, az=90)
        xy = sr.bounds.crs.point_offset_in_meters(xy, dy, az=180)

        gws.log.debug(f'get_features: {ox=} {oy=} {dx=} {dy=} {xy=}')

        point = gws.lib.shape.from_xy(xy[0], xy[1], sr.crs)

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
        """Set the bounds, the pixel size and the resolution of the request.

        The values are taken from the ``BBOX``, ``WIDTH`` and ``HEIGHT``
        parameters. If ``DPI`` or ``MAP_RESOLUTION`` is given, the resolution
        is converted from that DPI.

        Args:
            sr: Service request.

        Raises:
            ``server.error.MissingParameterValue``: If ``BBOX`` is missing.
            ``server.error.InvalidParameterValue``: If the width or height is out of range.
        """
        b = sr.requested_bounds('BBOX')
        if not b:
            raise server.error.MissingParameterValue('BBOX')
        sr.bounds = b

        sr.pxSize = sr.int_param('WIDTH'), sr.int_param('HEIGHT')
        if not (1 <= sr.pxSize[0] <= self.maxPixelSize):
            raise server.error.InvalidParameterValue('WIDTH')
        if not (1 <= sr.pxSize[1] <= self.maxPixelSize):
            raise server.error.InvalidParameterValue('HEIGHT')

        wh = sr.bounds.crs.extent_size_in_meters(sr.bounds.extent)

        dpi = sr.int_param('DPI', default=0) or sr.int_param('MAP_RESOLUTION', default=0)
        if dpi:
            # honor the dpi setting - compute the scale with "their" dpi and convert to "our" resolution
            sr.resX = gws.gis.zoom.scale_to_res(gws.lib.uom.mm_to_px(1000.0 * wh[0] / sr.pxSize[0], dpi), sr.bounds.crs)
            sr.resY = gws.gis.zoom.scale_to_res(gws.lib.uom.mm_to_px(1000.0 * wh[1] / sr.pxSize[1], dpi), sr.bounds.crs)
        else:
            sr.resX = wh[0] / sr.pxSize[0]
            sr.resY = wh[1] / sr.pxSize[1]

        # @TODO: is this correct?
        sr.resolution = sr.resX

        gws.log.debug(
            f'set_size_and_resolution: {wh=} px={sr.pxSize} {dpi=} resX={sr.resX} resY={sr.resY} 1:{gws.gis.zoom.res_to_scale(sr.resolution, sr.bounds.crs)}'
        )

    def visible_layer_caps(self, sr, lcs: list[server.LayerCaps]) -> list[server.LayerCaps]:
        """Filter layer caps by the resolution of the request.

        Args:
            sr: Service request.
            lcs: Layer caps.

        Returns:
            The layer caps whose layers are visible at the request resolution.
        """
        return [lc for lc in lcs if min(lc.layer.resolutions) <= sr.resolution <= max(lc.layer.resolutions)]
