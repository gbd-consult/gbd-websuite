from typing import Optional, cast

import gws
import gws.lib.image
import gws.lib.mime
import gws.base.legend



gws.ext.new.legend('combined')


class Config(gws.base.legend.Config):
    """Combined legend."""

    layerUids: list[str]
    """Layers to combine in the legend."""


class Object(gws.base.legend.Object):
    layerUids: list[str]

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
