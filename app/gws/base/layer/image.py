"""Base image layer."""

import gws

from . import core, util


class Object(core.Object):
    """Base image layer"""

    canRenderBox = True
    canRenderXyz = True
    supportsVectorOws = True

    def render(self, lri):
        if not self.grabber:
            return
        if lri.type == gws.LayerRenderInputType.xyz:
            return gws.LayerRenderOutput(content=self.grabber.get_tile((lri.x, lri.y, lri.z)))
        if lri.type == gws.LayerRenderInputType.box:
            def get_box(bounds, width, height):
                return self.grabber.get_box(bounds.extent, width, height)

            content = util.generic_render_box(self, lri, get_box)
            return gws.LayerRenderOutput(content=content)
