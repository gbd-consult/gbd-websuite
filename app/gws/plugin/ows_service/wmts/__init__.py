import math

import gws
import gws.base.web.error
import gws.lib.extent
import gws.lib.gis
import gws.lib.image
import gws.lib.legend
import gws.lib.render
import gws.lib.units
import gws.lib.crs
import gws.types as t

from .. import core


@gws.ext.Config('ows.service.wmts')
class Config(core.ServiceConfig):
    """WMTS Service configuration"""
    pass


@gws.ext.Object('ows.service.wmts')
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
                type='xml',
                path=gws.dirname(__file__) + '/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
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
                crs=gws.lib.crs.get3857(),
                matrices=self._tile_matrices(gws.lib.crs.c3857_extent, 0, 16),
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
            tile = {
                'matrix_set_uid': rd.req.param('TILEMATRIXSET'),
                'matrix_uid': rd.req.param('TILEMATRIX'),
                'row': int(rd.req.param('TILEROW')),
                'col': int(rd.req.param('TILECOL')),
            }
        except:
            raise gws.base.web.error.BadRequest()

        lcs = self.layer_caps_list_from_request(rd, ['layer'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound()

        tm_crs, bbox = self._tile_to_bbox(**tile)
        crs = rd.project.map.crs
        bbox = gws.lib.extent.transform(bbox, tm_crs, crs)

        render_input = gws.MapRenderInput(
            background_color=None,
            items=[],
            view=gws.lib.render.view_from_bbox(
                crs=crs,
                bbox=bbox,
                out_size=(256, 256),
                out_size_unit='px',
                rotation=0,
                dpi=0)
        )

        for lc in lcs:
            render_input.items.append(gws.MapRenderInputItem(
                type=gws.MapRenderInputItemType.image_layer,
                layer=lc.layer))

        renderer = gws.lib.render.Renderer()
        for _ in renderer.run(render_input):
            pass

        out = renderer.output
        if out.items and out.items[0].image:
            content = out.items[0].image.to_bytes()
        else:
            content = gws.lib.image.PIXEL_PNG8

        if self.root.application.developer_option('ows.annotate_wmts'):
            img = gws.lib.image.from_bytes(content)
            text = f"{tile['matrix_uid']} {tile['row']} {tile['col']}\n{bbox[0]}\n{bbox[1]}\n{bbox[2]}\n{bbox[3]}"
            content = img.add_text(text, x=10, y=10).add_box().to_bytes()

        return gws.ContentResponse(mime='image/png', content=content)

    def handle_getlegendgraphic(self, rd: core.Request):
        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'], self.SCOPE_LEAF)
        if not lcs:
            raise gws.base.web.error.NotFound('No layers found')
        out = gws.lib.legend.render(gws.Legend(layers=[lc.layer for lc in lcs if lc.has_legend]))
        return gws.ContentResponse(mime='image/png', content=gws.lib.legend.to_bytes(out) or gws.lib.image.PIXEL_PNG8)

    ##

    def _tile_to_bbox(self, matrix_set_uid, matrix_uid, row, col):
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

        w, h = gws.lib.extent.size(tm.extent)
        span = w / tm.width

        x = tm.x + col * span
        y = tm.y - row * span

        bbox = x, y - span, x + span, y
        return tms.crs, bbox

    def _tile_matrices(self, extent, min_zoom, max_zoom, tile_size=256):
        ms = []

        # north origin
        extent = extent[0], extent[3], extent[2], extent[1]

        w, h = gws.lib.extent.size(extent)

        for z in range(min_zoom, max_zoom + 1):
            size = 1 << z
            res = w / (tile_size * size)
            ms.append(gws.TileMatrix(
                uid='%02d' % z,
                scale=gws.lib.units.res2scale(res),
                x=extent[0],
                y=extent[1],
                tile_width=tile_size,
                tile_height=tile_size,
                width=size,
                height=size,
                extent=extent,
            ))

        return ms
