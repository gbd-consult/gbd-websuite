import gws
import gws.lib.image
import gws.lib.mime
import gws.base.legend


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
        lros = []
        for uid in self.layerUids:
            layer = self.root.find(gws.ext.object.layer, uid)
            if layer:
                lro = layer.render_legend(args)
                if lro:
                    lros.append(lro)

        return gws.base.legend.combine_outputs(lros, self.options)
