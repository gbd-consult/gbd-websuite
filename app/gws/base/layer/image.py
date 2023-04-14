import gws
import gws.gis.extent
import gws.lib.image
import gws.gis.mpx as mpx

from . import core



class Object(core.Object):
    """Base image layer"""

    canRenderBox = True
    canRenderXyz = True
    canRenderSvg = False
    canRasterOws = True


    def configure(self):
        self.configure_base()




    def props(self, user):
        p = super().props(user)


        return p

    def render_box(self, view, extra_params=None):
        uid = self.uid
        if not self.has_cache:
            uid += '_NOCACHE'

        if not view.rotation:
            return gws.gis.mpx.wms_request(uid, view.bounds, view.pxSize[0], view.pxSize[1], forward=extra_params)

        # rotation: render a circumsquare around the wanted extent

        circ = gws.gis.extent.circumsquare(view.bounds.extent)
        w, h = view.pxSize
        d = gws.gis.extent.diagonal((0, 0, w, h))

        r = gws.gis.mpx.wms_request(
            uid,
            gws.Bounds(crs=view.bounds.crs, extent=circ),
            width=d,
            height=d,
            forward=extra_params)
        if not r:
            return

        img = gws.lib.image.from_bytes(r)

        # rotate the square (NB: PIL rotations are counter-clockwise)
        # and crop the square back to the wanted extent

        img.rotate(-view.rotation).crop((
            d / 2 - w / 2,
            d / 2 - h / 2,
            d / 2 + w / 2,
            d / 2 + h / 2,
        ))

        return img.to_bytes()

    def render_xyz(self, x, y, z):
        return gws.gis.mpx.wmts_request(
            self.uid,
            x, y, z,
            tile_matrix=self.grid_uid,
            tile_size=self.grid.tileSize)

