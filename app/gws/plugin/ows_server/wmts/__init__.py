"""WMTS Service.

Implements WMTS 1.0.0.
This implementation only supports ``GET`` requests with ``KVP`` encoding.

References:
    - OGC 07-057r7 (https://portal.ogc.org/files/?artifact_id=35326)
"""

from typing import Optional

import gws
import gws.config.util
import gws.base.ows.server as server
import gws.lib.crs
import gws.lib.grid
import gws.lib.mime
import gws.gis.render
import gws.gis.zoom

gws.ext.new.owsService('wmts')

MAX_LEVEL = 20
"""Finest advertised tile matrix level."""


class Config(server.service.Config):
    """WMTS Service configuration"""

    grids: Optional[list[gws.lib.grid.Config]]
    """Tile matrix grids, one per CRS. A supported CRS without a grid uses the default grid."""


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
    grids: dict[str, gws.MapGrid]
    """Grids by tile matrix set identifier."""

    def configure(self):
        gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

        configured = {}
        for p in self.cfg('grids', default=[]):
            if not p.crs:
                raise gws.ConfigurationError('wmts grid: crs is required')
            crs = gws.lib.crs.require(p.crs)
            if crs.srid in configured:
                raise gws.ConfigurationError(f'wmts grid: duplicate crs {crs.srid}')
            configured[crs.srid] = gws.lib.grid.new(
                gws.lib.grid.Options(
                    crs=crs,
                    extent=p.extent,
                    baseResolution=p.baseResolution,
                    tileSize=p.tileSize,
                    withSnap=p.withSnap,
                )
            )

        supported = {b.crs.srid for b in self.supportedBounds}
        for srid in configured:
            if srid not in supported:
                raise gws.ConfigurationError(f'wmts grid: crs {srid} is not supported by the service')

        # @TODO different matrix sets per layer
        self.tileMatrixSets = []
        self.grids = {}
        for b in self.supportedBounds:
            # see https://docs.opengeospatial.org/is/13-082r2/13-082r2.html#29
            mg = configured.get(b.crs.srid) or gws.lib.grid.for_crs(b.crs)
            ident = f'TMS_{b.crs.srid}'
            self.grids[ident] = mg
            self.tileMatrixSets.append(
                gws.TileMatrixSet(
                    identifier=ident,
                    crs=b.crs,
                    matrices=self.make_tile_matrices(mg, 0, MAX_LEVEL),
                )
            )

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

    def make_tile_matrices(self, mg: gws.MapGrid, min_zoom, max_zoom):
        ms = []

        for z in range(min_zoom, max_zoom + 1):
            nx, ny = gws.lib.grid.tile_count_for_level(mg, z)
            res = gws.lib.grid.resolution_for_level(mg, z)
            ms.append(
                gws.TileMatrix(
                    identifier=f'{z:02d}',
                    scale=gws.gis.zoom.res_to_scale(res, mg.crs),
                    resolution=res,
                    x=mg.extent[0],
                    y=mg.extent[3],
                    tileWidth=mg.tileSize,
                    tileHeight=mg.tileSize,
                    width=nx,
                    height=ny,
                    extent=mg.extent,
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

        tms_uid = sr.string_param('TILEMATRIXSET')
        tm_uid = sr.string_param('TILEMATRIX')
        row = sr.int_param('TILEROW')
        col = sr.int_param('TILECOL')

        bounds = self.bounds_for_tile(tms_uid, tm_uid, row, col)
        if not bounds:
            raise server.error.TileOutOfRange()
        gws.log.debug(f'WMTS: bounds for tile {tms_uid=} {tm_uid=} {row=} {col=}: {bounds}')

        mime = sr.requested_format('FORMAT')
        ts = self.grids[tms_uid].tileSize

        mri = gws.MapRenderInput(
            backgroundColor=None,
            bbox=bounds.extent,
            targetCrs=bounds.crs,
            mapSize=(ts, ts, gws.Uom.px),
            planes=[
                gws.MapRenderInputPlane(
                    type=gws.MapRenderInputPlaneType.imageLayer,
                    layer=lc.layer,
                )
                for lc in lcs
            ],
        )

        mro = gws.gis.render.render_map(mri)

        if self.root.app.developer_option('ows.annotate_wmts'):
            e = bounds.extent
            text = f'{tm_uid} {row} {col}\n{e[0]}\n{e[1]}\n{e[2]}\n{e[3]}'
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

    def bounds_for_tile(self, tms_uid, tm_uid, row, col):
        tms = self.get_matrix_set(tms_uid)
        if not tms:
            return
        tm = self.get_matrix(tms, tm_uid)
        if not tm:
            return

        mg = self.grids[tms_uid]
        z = int(tm.identifier)
        return gws.Bounds(crs=tms.crs, extent=gws.lib.grid.extent_for_tile(mg, (col, row, z)))

    def get_matrix_set(self, tms_uid):
        for tms in self.tileMatrixSets:
            if tms.identifier == tms_uid:
                return tms

    def get_matrix(self, tms: gws.TileMatrixSet, tm_uid):
        for tm in tms.matrices:
            if tm.identifier == tm_uid:
                return tm
