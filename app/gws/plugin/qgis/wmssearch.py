"""Qgis/WMS automatic search provider"""

import gws
import gws.base.search
import gws.types as t

from . import provider


gws.ext.new.finder('qgiswms')

class Object(gws.base.search.provider.Object):
    supports_geometry = True

    provider: provider.Object
    source_layers: list[gws.SourceLayer]

    def configure(self):
        self.provider = self.cfg('_provider')
        self.source_layers = self.cfg('_sourceLayers')

    def can_run(self, args):
        return (
                super().can_run(args)
                and args.shapes
                and len(args.shapes) == 1
                and args.shapes[0].geometry_type == gws.GeometryType.point)

    def run(self, args: gws.SearchArgs, layer: gws.ILayer = None) -> list[gws.IFeature]:
        return self.provider.find_features(args, self.source_layers)
