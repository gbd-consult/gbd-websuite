"""Raster layer grabber."""

import gws
import gws.base.grabber.box
import gws.lib.mapserver.core


class Object(gws.base.grabber.box.Object):
    """Grabber that renders boxes with an in-process MapServer map."""

    msOptions: gws.MapServerLayerOptions
    """MapServer layer options."""

    def __init__(self, opts: gws.base.grabber.Options, msOptions: gws.MapServerLayerOptions):
        """Create the grabber.

        Args:
            opts: Grabber options.
            msOptions: MapServer layer options.
        """
        super().__init__(opts)
        self.msOptions = msOptions
        self.sourceCrs = self.targetCrs
        self.maxRequestPixels = 9000

    def fetch_box_as_image(self, bounds, w, h, params=None):
        ms_map = gws.lib.mapserver.core.new_map()
        ms_map.add_layer(self.msOptions)
        return ms_map.draw(bounds, (w, h))

    def fetch_box_as_bytes(self, bounds, w, h, params=None):
        return self.to_bytes(self.fetch_box_as_image(bounds, w, h, params))
