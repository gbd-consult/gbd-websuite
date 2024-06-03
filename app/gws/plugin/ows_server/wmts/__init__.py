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
import gws.gis.crs
import gws.gis.extent
import gws.lib.image
import gws.lib.mime
import gws.gis.render
import gws.lib.uom as units

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

    tileMatrixSets: list[gws.TileMatrixSet]

    def configure(self):
        gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

        # @TODO different matrix sets per layer
        self.tileMatrixSets = [
            # see https://docs.opengeospatial.org/is/13-082r2/13-082r2.html#29
            gws.TileMatrixSet(
                uid=f'TMS_{b.crs.srid}',
                crs=b.crs,
                matrices=self._tile_matrices(b.extent, 0, 16),
            )
            for b in self.supportedBounds
        ]

    def configure_operations(self):
        self.supportedOperations = [
            gws.OwsOperation(verb=gws.OwsVerb.GetCapabilities, formats=self.template_formats(gws.OwsVerb.GetCapabilities)),
            gws.OwsOperation(verb=gws.OwsVerb.GetLegendGraphic, formats=self.imageFormats),
            gws.OwsOperation(verb=gws.OwsVerb.GetTile, formats=self.imageFormats),
        ]

    def activate(self):
        self.handlers = {
            gws.OwsVerb.GetCapabilities: self.handle_get_capabilities,
            gws.OwsVerb.GetLegendGraphic: self.handle_get_legend_graphic,
            gws.OwsVerb.GetTile: self.handle_get_tile,
        }

    ##

    def layer_is_suitable(self, layer: gws.Layer):
        return layer.canRenderBox

    def requested_layer_caps(self, sr: server.request.Object):
        lcs = []

        for name in sr.list_param('layer'):
            for lc in sr.layerCapsList:
                if not server.layer_caps.layer_name_matches(lc, name):
                    continue
                if lc.isGroup:
                    lcs.extend(lc.leaves)
                else:
                    lcs.append(lc)

        if not lcs:
            raise gws.base.web.error.NotFound('Layer not found')

        return gws.u.uniq(lcs)

    ##

    def handle_get_capabilities(self, sr: server.request.Object):
        return self.template_response(
            sr,
            format=sr.string_param('format', default=''),
            layerCapsList=sr.layerCapsList,
            tileMatrixSets=self.tileMatrixSets,
        )

    def handle_get_tile(self, sr: server.request.Object):
        lcs = self.requested_layer_caps(sr)
        if len(lcs) != 1:
            raise gws.base.web.error.BadRequest(f'Invalid LAYER parameter')

        matrix_set_uid = sr.string_param('tileMatrixSet')
        matrix_uid = sr.string_param('tileMatrix')
        row = sr.int_param('tileRow')
        col = sr.int_param('tileCol')

        bounds = self._bounds_for_tile(matrix_set_uid, matrix_uid, row, col)
        if not bounds:
            raise gws.base.web.error.BadRequest()

        mri = gws.MapRenderInput(
            backgroundColor=None,
            bbox=bounds.extent,
            crs=bounds.crs,
            mapSize=(256, 256, gws.Uom.px),
            planes=[
                gws.MapRenderInputPlane(type=gws.MapRenderInputPlaneType.imageLayer, layer=lc.layer)
                for lc in lcs
            ]
        )

        mro = gws.gis.render.render_map(mri)

        if mro.planes and mro.planes[0].image:
            content = mro.planes[0].image.to_bytes()
        else:
            content = gws.lib.image.PIXEL_PNG8

        if self.root.app.developer_option('ows.annotate_wmts'):
            img = gws.lib.image.from_bytes(content)
            e = bounds.extent
            text = f"{matrix_uid} {row} {col}\n{e[0]}\n{e[1]}\n{e[2]}\n{e[3]}"
            content = img.add_text(text, x=10, y=10).add_box().to_bytes()

        return gws.ContentResponse(mime=gws.lib.mime.PNG, content=content)

    def handle_get_legend_graphic(self, sr: server.request.Object):
        lcs = self.requested_layer_caps(sr)
        return sr.render_legend(lcs)

    ##

    def _bounds_for_tile(self, matrix_set_uid, matrix_uid, row, col):
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

        w, h = gws.gis.extent.size(tm.extent)
        span = w / tm.width

        x = tm.x + col * span
        y = tm.y - row * span

        bbox = x, y - span, x + span, y
        return gws.Bounds(crs=tms.crs, extent=bbox)

    def _tile_matrices(self, extent, min_zoom, max_zoom, tile_size=256):
        ms = []

        # north origin
        extent = extent[0], extent[3], extent[2], extent[1]

        w, h = gws.gis.extent.size(extent)

        for z in range(min_zoom, max_zoom + 1):
            size = 1 << z
            res = w / (tile_size * size)
            ms.append(gws.TileMatrix(
                uid=f'{z:02d}',
                scale=gws.lib.uom.res_to_scale(res),
                x=extent[0],
                y=extent[1],
                tileWidth=tile_size,
                tileHeight=tile_size,
                width=size,
                height=size,
                extent=extent,
            ))

        return ms
