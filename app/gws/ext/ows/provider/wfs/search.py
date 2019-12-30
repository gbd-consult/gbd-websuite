import gws
import gws.common.search.provider
import gws.gis.util
import gws.gis.proj
import gws.gis.shape

import gws.types as t

from . import provider, types, util

class Config(gws.common.search.provider.Config, types.WfsServiceConfig):
    pass


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.invert_axis_crs = []
        self.provider: provider.Object = None
        self.source_layers: t.List[types.SourceLayer] = []
        self.url = ''

    def configure(self):
        super().configure()
        util.configure_wfs(self)

    def can_run(self, args):
        return (
                self.provider.operation('GetFeature')
                and args.shapes
                and not args.keyword
        )

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        shape = gws.gis.shape.union(args.shapes)
        if shape.type == 'Point':
            shape = shape.tolerance_buffer(args.get('tolerance'))

        fs = util.find_features(self, shape.bounds, args.crs, args.limit)

        # @TODO excluding geometryless features for now
        fs = [f for f in fs if f.shape and f.shape.intersects(shape)]
        gws.log.debug(f'WFS_QUERY: AFTER FILTER: {len(fs)}')

        return fs
