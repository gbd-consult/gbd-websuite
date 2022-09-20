import math

import gws
import gws.base.web.error
import gws.gis.crs
import gws.gis.extent
import gws.lib.image
import gws.lib.mime
import gws.gis.render
import gws.lib.units as units
import gws.types as t

from .. import core


@gws.ext.config.owsService('wmts')
class Config(core.ServiceConfig):
    """WMTS Service configuration"""
    pass


@gws.ext.object.owsService('wmts')
class Object(core.Service):
    protocol = gws.OwsProtocol.WMTS
    supported_versions = ['1.0.0']
    is_raster_ows = True

    tile_matrix_sets: t.List[gws.TileMatrixSet]

    @property
    def service_link(self):
        if self.project:
            return gws.Data(url=self.service_url_path(self.project), scheme='OGC:WMTS', function='search')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='py',
                path=gws.dirname(__file__) + '/templates/getCapabilities.py',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
                access='all:allow',
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
        # @TODO more crs
        # @TODO different matrix sets per layer
        self.tile_matrix_sets = [
            # see https://docs.opengeospatial.org/is/13-082r2/13-082r2.html#29
            gws.TileMatrixSet(
                uid='EPSG_3857',
                crs=gws.gis.crs.get3857(),
                matrices=self._tile_matrices(gws.gis.crs.CRS_3857_EXTENT, 0, 16),
            )
        ]

    ##

    def handle_getcapabilities(self, rd: core.Request):
        tree = self.layer_caps_tree(rd)
        return self.template_response(rd, gws.OwsVerb.GetCapabilities, context={
            'layer_caps_list': tree.leaves,
            'version': self.request_version(rd),
            'tile_matrix_sets': self.tile_matrix_sets
        })

    def handle_gettile(self, rd: core.Request):
        try:
            matrix_set_uid = rd.req.param('TILEMATRIXSET')
            matrix_uid = rd.req.param('TILEMATRIX')
            row = int(rd.req.param('TILEROW'))
            col = int(rd.req.param('TILECOL'))
        except Exception:
            raise gws.base.web.error.BadRequest()

        lcs = self.layer_caps_list_from_request(rd, ['layer'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound()

        bounds = self._bounds_for_tile(matrix_set_uid, matrix_uid, row, col)
        if not bounds:
            raise gws.base.web.error.BadRequest()

        # crs = rd.project.map.crs
        # bbox = gws.gis.extent.transform(bbox, tm_crs, crs)

        mri = gws.MapRenderInput(
            background_color=None,
            bbox=bounds.extent,
            crs=bounds.crs,
            out_size=(256, 256, units.PX),
            planes=[
                gws.MapRenderInputPlane(type='image_layer', layer=lc.layer)
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

    def handle_getlegendgraphic(self, rd: core.Request):
        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')
        out = gws.gis.legend.render(gws.Legend(layers=[lc.layer for lc in lcs if lc.has_legend]))
        return gws.ContentResponse(mime=gws.lib.mime.PNG, content=gws.gis.legend.to_bytes(out) or gws.lib.image.PIXEL_PNG8)

    ##

    def _bounds_for_tile(self, matrix_set_uid, matrix_uid, row, col):
        tms = None
        tm = None

        for m in self.tile_matrix_sets:
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
                uid='%02d' % z,
                scale=gws.lib.units.res_to_scale(res),
                x=extent[0],
                y=extent[1],
                tile_width=tile_size,
                tile_height=tile_size,
                width=size,
                height=size,
                extent=extent,
            ))

        return ms
