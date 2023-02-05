import gws
import gws.lib.image
import gws.lib.mime
import gws.types as t

from ... import core

gws.ext.new.legend('combined')


class Config(core.Config):
    """Combined legend."""

    layerUids: t.List[str]
    """layers"""


class Object(core.Object):
    layerUids: t.List[str]

    def configure(self):
        self.layerUids = self.var('layerUids')

    def render(self, args=None):
        lros = []
        for uid in self.layerUids:
            layer = self.root.find(gws.ext.object.layer, uid)
            if layer:
                lro = layer.render_legend(args)
                if lro:
                    lros.append(lro)

        return core.combine_outputs(lros, self.options)
