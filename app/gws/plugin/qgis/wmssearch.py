"""Qgis/WMS automatic search provider"""

import gws
import gws.base.search
import gws.types as t

from . import provider


@gws.ext.object.finder('qgiswms')
class Object(gws.base.search.provider.Object):
    supports_geometry = True

    provider: provider.Object
    source_layers: t.List[gws.SourceLayer]

    def configure(self):
        self.provider = self.var('_provider')
        self.source_layers = self.var('_source_layers')

    def can_run(self, args):
        return (
                super().can_run(args)
                and args.shapes
                and len(args.shapes) == 1
                and args.shapes[0].geometry_type == gws.GeometryType.point)

    def run(self, args: gws.SearchArgs, layer: gws.ILayer = None) -> t.List[gws.IFeature]:
        return self.provider.find_features(args, self.source_layers)