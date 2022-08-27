import gws
import gws.lib.image
import gws.lib.mime
import gws.types as t

from .. import main


@gws.ext.config.legend('combined')
class Config(main.Config):
    """Combined legend."""

    layerUids: t.List[str]  #: layers


@gws.ext.object.legend('combined')
class Object(main.Object):
    layerUids: t.List[str]

    def configure(self):
        super().configure()
        self.layerUids = self.var('layerUids')

    def render(self, args=None):
        lros = []
        for uid in self.layersUids:
            layer = self.root.find(gws.ext.object.layer, uid)
            if layer:
                lro = layer.render_legend(args)
                if lro:
                    lros.append(lro)

        return main.combine_outputs(lros, self.options)
