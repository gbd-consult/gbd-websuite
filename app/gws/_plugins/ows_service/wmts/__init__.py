import io
import math

from gws.ext.ows.provider.wmts.types import TileMatrix, TileMatrixSet

import gws
import gws.types as t
import gws.lib.metadata
import gws.base.ows.service as ows
import gws.lib.extent
import gws.lib.gml
import gws.lib.legend
import gws.lib.proj
import gws.lib.render
import gws.lib.shape
import gws.lib.mime
import gws.lib.misc
import gws.lib.os2
import gws.lib.units
import gws.lib.xml2
import gws.base.web.error


class Config(gws.base.ows.service.Config):
    """WMTS Service configuration"""
    pass


class Object(ows.Base):

    @property
    def service_link(self):
        if self.project:
            return gws.MetaLink(url=self.url_for_project(self.project), scheme='OGC:WMTS', function='search')

    @property
    def default_templates(self):
        return [
            gws.Config(
                type='xml',
                path=gws.APP_DIR + '/gws/ext/ows/service/wmts/templates/getCapabilities.cx',
                subject='ows.GetCapabilities',
                mimeTypes=['xml'],
            ),
        ]

    @property
    def default_metadata(self):
        return gws.Data(
            inspireDegreeOfConformity=gws.MetaInspireDegreeOfConformity.notEvaluated,
            inspireMandatoryKeyword=gws.MetaInspireMandatoryKeyword.infoMapAccessService,
            inspireResourceType=gws.MetaInspireResourceType.service,
            inspireSpatialDataServiceType=gws.MetaInspireSpatialDataServiceType.view,
            isoScope=gws.MetaIsoScope.dataset,
            isoSpatialRepresentationType=gws.MetaIsoSpatialRepresentationType.vector,
        )

    @property
    def default_name(self):
        return 'WMTS'

    ##

    def configure(self):
        

        self.type = 'wmts'
        self.supported_versions = ['1.0.0']

        # @TODO more crs
        self.matrix_sets = [
            # see https://docs.opengeospatial.org/is/13-082r2/13-082r2.html#29
            TileMatrixSet(
                uid='EPSG_3857',
                crs='EPSG:3857',
                matrices=_tile_matrices(EPSG3857_EXTENT, 0, 16),
            )
        ]

    ##

    def handle_getcapabilities(self, rd: ows.Request):
        root = self.layer_root_caps(rd)
        if not root:
            gws.log.debug(f'service={self.uid!r}: no layer_root_caps')
            raise gws.base.web.error.NotFound()

        return self.template_response(rd, 'GetCapabilities', context={
            'layer_root_caps': root,
            'version': self.request_version(rd),
            'matrix_sets': self.matrix_sets
        })

    def handle_gettile(self, rd: ows.Request):
        try:
            tile = gws.Data(
                matrix_set_uid=rd.req.param('TILEMATRIXSET'),
                matrix_uid=rd.req.param('TILEMATRIX'),
                row=int(rd.req.param('TILEROW')),
                col=int(rd.req.param('TILECOL')),
            )
        except:
            raise gws.base.web.error.BadRequest()

        lcs = self.layer_caps_list_from_request(rd, ['layer'])
        if not lcs:
            raise gws.base.web.error.NotFound()

        tm_crs, bbox = self._tile_to_bbox(tile)
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
        if not out.items:
            img = gws.lib.misc.Pixels.png8
        else:
            buf = io.BytesIO()
            out.items[0].image.save(buf, format='png')
            img = buf.getvalue()

        return gws.ContentResponse(mime='image/png', content=img)

    def handle_getlegendgraphic(self, rd: ows.Request):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        lcs = self.layer_caps_list_from_request(rd, ['layer', 'layers'])
        if not lcs:
            raise gws.base.web.error.NotFound()

        paths = [lc.layer.render_legend() for lc in lcs if lc.has_legend]
        out = gws.lib.legend.combine_legend_paths(paths)
        return gws.ContentResponse(mime='image/png', content=out or gws.lib.misc.Pixels.png8)

    ##

    def _tile_to_bbox(self, tile):
        tms = None
        tm = None

        for m in self.matrix_sets:
            if m.uid == tile.matrix_set_uid:
                tms = m

        if not tms:
            return

        for m in tms.matrices:
            if m.uid == tile.matrix_uid:
                tm = m

        if not tm:
            return

        w, h = gws.lib.extent.size(tm.extent)
        span = w / tm.width

        x = tm.x + tile.col * span
        y = tm.y - tile.row * span

        bbox = x, y, x + span, y - span
        return tms.crs, bbox


EPSG3857_RADIUS = 6378137

EPSG3857_EXTENT = [
    -math.pi * EPSG3857_RADIUS, math.pi * EPSG3857_RADIUS,
    math.pi * EPSG3857_RADIUS, -math.pi * EPSG3857_RADIUS
]


def _tile_matrices(extent, min_zoom, max_zoom, tile_size=256):
    ms = []
    w, h = gws.lib.extent.size(extent)

    for z in range(min_zoom, max_zoom + 1):
        size = 1 << z
        res = w / (tile_size * size)
        ms.append(TileMatrix(
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
