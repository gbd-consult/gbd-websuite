"""WMS grabber."""

import gws
import gws.base.grabber.box
import gws.lib.extent
import gws.lib.gdalx
import gws.lib.image

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    sourceCrs: gws.Crs

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.sourceLayers = self.cfg('_defaultSourceLayers')
        self.sourceCrs = self.cfg('_defaultSourceCrs')

    def fetch_box(self, extent, width, height):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        mpx = self.serviceProvider.maxRequestPixels
        rw = min(w, mpx)
        rh = min(h, mpx)

        if self.targetCrs == self.sourceCrs:
            blob = self.serviceProvider.get_map(
                gws.Bounds(crs=self.sourceCrs, extent=extent),
                rw,
                rh,
                self.sourceLayers,
                self.mime,
            )
            img = gws.lib.image.from_size((rw, rh))
            img.paste(gws.lib.image.from_bytes(blob), (0, 0))
            if rw != w or rh != h:
                img.resize((w, h))
            return img

        src_extent = gws.lib.extent.transform(extent, self.targetCrs, self.sourceCrs)
        src_res = (src_extent[2] - src_extent[0]) / rw
        src_extent = gws.lib.extent.buffer(src_extent, src_res * 2)
        sw = rw + 4
        sh = rh + 4

        blob = self.serviceProvider.get_map(
            gws.Bounds(crs=self.sourceCrs, extent=src_extent),
            sw,
            sh,
            self.sourceLayers,
            self.mime,
        )

        canvas = gws.lib.image.from_size((sw, sh))
        canvas.paste(gws.lib.image.from_bytes(blob), (0, 0))

        with gws.lib.gdalx.open_from_image(canvas, gws.Bounds(crs=self.sourceCrs, extent=src_extent)) as ds:
            img = ds.warp_to_image(
                dict(
                    dstSRS=self.targetCrs.epsg,
                    outputBounds=extent,
                    outputBoundsSRS=self.targetCrs.epsg,
                    width=w,
                    height=h,
                    resampleAlg='bilinear',
                )
            )

        return img
