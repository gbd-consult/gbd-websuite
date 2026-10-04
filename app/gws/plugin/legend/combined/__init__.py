"""Combined legend.

Renders the legends of several layers, given by ``layerUids``, and combines
them into one image. Layers that are not found or have no legend are skipped.

Example::

    legend {
        type "combined"
        layerUids ["layer_1", "layer_2"]
    }
"""

from typing import Optional, cast

import gws
import gws.lib.image
import gws.lib.mime
import gws.base.legend


@gws.ext.config.legend('combined')
class Config(gws.base.legend.Config):
    """Legend combining the legends of several layers."""

    layerUids: list[str]
    """UIDs of the layers whose legends are combined."""


@gws.ext.object.legend('combined')
class Object(gws.base.legend.Object):
    """Combined legend."""

    layerUids: list[str]
    """UIDs of the layers whose legends are combined."""

    def configure(self):
        self.layerUids = self.cfg('layerUids')

    def render(self, args=None):
        outputs = []

        for uid in self.layerUids:
            layer = cast(gws.Layer, self.root.get(uid, gws.ext.object.layer))
            if layer and layer.legend:
                lro = layer.legend.render(args)
                if lro:
                    outputs.append(lro)

        return gws.base.legend.combine_outputs(outputs, self.options)
