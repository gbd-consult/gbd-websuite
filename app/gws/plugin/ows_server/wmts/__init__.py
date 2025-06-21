"""WMTS Service.

Implements WMTS 1.0.0.
This implementation only supports ``GET`` requests with ``KVP`` encoding.

References:
    - OGC 07-057r7 (https://portal.ogc.org/files/?artifact_id=35326)
"""

import gws
import gws.config.util
import gws.base.ows.server as server
import gws.base.web
import gws.lib.crs
import gws.lib.extent
import gws.lib.image
import gws.lib.mime
import gws.gis.render
import gws.lib.uom

gws.ext.new.owsService('wmts')


class Config(server.service.Config):
    """WMTS Service configuration"""

    pass


_DEFAULT_TEMPLATES = [
    gws.Config(
        type='py',
        path=gws.u.dirname(__file__) + '/templates/getCapabilities.cx.py',
        subject='ows.GetCapabilities',
        mimeTypes=[gws.lib.mime.XML],
        access=gws.c.PUBLIC,
    ),
]

_DEFAULT_METADATA = gws.Metadata(
    inspireDegreeOfConformity='notEvaluated',
    inspireMandatoryKeyword='infoMapAccessService',
    inspireResourceType='service',
    inspireSpatialDataServiceType='view',
    isoScope='dataset',
    isoSpatialRepresentationType='vector',
)


class Object(server.service.Object):
    protocol = gws.OwsProtocol.WMTS
    supportedVersions = ['1.0.0']
    isRasterService = True
    isOwsCommon = True

    tileMatrixSets: list[gws.TileMatrixSet]

    def configure(self):
        gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

        # @TODO different matrix sets per layer
        self.tileMatrixSets = [
            # see https://docs.opengeospatial.org/is/13-082r2/13-082r2.html#29
            gws.TileMatrixSet(
                uid=f'TMS_{b.crs.srid}',
                crs=b.crs,
                matrices=self.make_tile_matrices(b.extent, 0, 16),
            )
            for b in self.supportedBounds
        ]

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(
                verb=gws.OwsVerb.GetCapabilities,
                formats=self.available_formats(gws.OwsVerb.GetCapabilities),
                handlerName='handle_get_capabilities',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetLegendGraphic,
                formats=self.available_formats(gws.OwsVerb.GetLegendGraphic),
                handlerName='handle_get_legend_graphic',
            ),
            gws.OwsOperation(
                verb=gws.OwsVerb.GetTile,
                formats=self.available_formats(gws.OwsVerb.GetTile),
                handlerName='handle_get_tile',
            ),
        ]

    def make_tile_matrices(self, extent, min_zoom, max_zoom, tile_size=256):
        ms = []

        # north origin
        extent = extent[0], extent[3], extent[2], extent[1]

        w, h = gws.lib.extent.size(extent)

        for z in range(min_zoom, max_zoom + 1):
            size = 1 << z
            res = w / (tile_size * size)
            ms.append(
                gws.TileMatrix(
                    uid=f'{z:02d}',
                    scale=gws.lib.uom.res_to_scale(res),
                    x=extent[0],
                    y=extent[1],
                    tileWidth=tile_size,
                    tileHeight=tile_size,
                    width=size,
                    height=size,
                    extent=extent,
                )
            )

        return ms

    ##

    def init_request(self, req):
        sr = super().init_request(req)
        sr.require_project()
        return sr

    def layer_is_compatible(self, layer: gws.Layer):
        return not layer.isGroup and layer.canRenderBox

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(
            sr,
            sr.requested_format('FORMAT'),
            layerCapsList=sr.layerCapsList,
            tileMatrixSets=self.tileMatrixSets,
        )

    def handle_get_tile(self, sr: server.request.Object):
        lcs = self.requested_layer_caps(sr)
        if len(lcs) != 1:
            raise server.error.InvalidParameterValue('LAYER')

        matrix_set_uid = sr.string_param('TILEMATRIXSET')
        matrix_uid = sr.string_param('TILEMATRIX')
        row = sr.int_param('TILEROW')
        col = sr.int_param('TILECOL')

        bounds = self.bounds_for_tile(matrix_set_uid, matrix_uid, row, col)
        if not bounds:
            raise server.error.TileOutOfRange()

        mime = sr.requested_format('FORMAT')

        mri = gws.MapRenderInput(
            backgroundColor=None,
            bbox=bounds.extent,
            crs=bounds.crs,
            mapSize=(256, 256, gws.Uom.px),
            planes=[gws.MapRenderInputPlane(type=gws.MapRenderInputPlaneType.imageLayer, layer=lc.layer) for lc in lcs],
        )

        mro = gws.gis.render.render_map(mri)

        if self.root.app.developer_option('ows.annotate_wmts'):
            e = bounds.extent
            text = f'{matrix_uid} {row} {col}\n{e[0]}\n{e[1]}\n{e[2]}\n{e[3]}'
            mro.planes[0].image = mro.planes[0].image.add_text(text, x=10, y=10).add_box()

        return self.image_response(sr, mro.planes[0].image, mime)

    def handle_get_legend_graphic(self, sr: server.request.Object):
        lcs = self.requested_layer_caps(sr)
        return self.render_legend(sr, lcs, sr.requested_format('FORMAT'))

    ##

    def requested_layer_caps(self, sr: server.request.Object):
        lcs = []

        for name in sr.list_param('LAYER'):
            for lc in sr.layerCapsList:
                if not server.layer_caps.layer_name_matches(lc, name):
                    continue
                lcs.append(lc)

        if not lcs:
            raise server.error.LayerNotDefined()

        return gws.u.uniq(lcs)

    def bounds_for_tile(self, matrix_set_uid, matrix_uid, row, col):
        tms = None
        tm = None

        for m in self.tileMatrixSets:
            if m.uid == matrix_set_uid:
                tms = m

        if not tms:
            return

        for m in tms.matrices:
            if m.uid == matrix_uid:
                tm = m

        if not tm:
            return

        w, h = gws.lib.extent.size(tm.extent)
        span = w / tm.width

        x = tm.x + col * span
        y = tm.y - row * span

        bbox = x, y - span, x + span, y
        return gws.Bounds(crs=tms.crs, extent=bbox)
