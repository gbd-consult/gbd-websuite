import gws
import gws.lib.image
import gws.lib.mime
import gws.base.legend

import gws.types as t


gws.ext.new.legend('combined')


class Config(gws.base.legend.Config):
    """Combined legend."""

    layerUids: list[str]
    """layers"""


class Object(gws.base.legend.Object):
    layerUids: list[str]

    def configure(self):
        self.layerUids = self.cfg('layerUids')

    def render(self, args=None):
        outputs = []

        for uid in self.layerUids:
            layer = t.cast(gws.ILayer, self.root.get(uid, gws.ext.object.layer))
            if layer and layer.legend:
                lro = layer.legend.render(args)
                if lro:
                    outputs.append(lro)

        return gws.base.legend.combine_outputs(outputs, self.options)
