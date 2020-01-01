import gws
import gws.common.search.provider
import gws.common.ows.provider
import gws.common.map
import gws.gis.util
import gws.types as t

from . import provider, util

class Config(gws.common.search.provider.Config, util.WmsConfig):
    pass


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.invert_axis_crs = []
        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''
        self.map: gws.common.map.Object = None

    def configure(self):
        super().configure()
        self.map = self.get_closest('gws.common.map')
        util.configure_wms(self, queryable_only=True)

    def can_run(self, args):
        return (
                self.source_layers
                and self.provider.operation('GetFeatureInfo')
                and args.shapes
                and args.shapes[0].type == 'Point'
                and not args.keyword
        )

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        shape = args.shapes[0]
        crs = gws.gis.util.best_crs(args.crs, self.provider.supported_crs)
        shape = shape.transform(crs)
        axis = gws.gis.util.best_axis(args.crs, self.invert_axis_crs, 'WMS', self.provider.version)

        fa = t.SearchArgs({
            'axis': axis,
            'bbox': '',
            'count': args.limit,
            'crs': crs,
            'layers': [sl.name for sl in self.source_layers],
            'params': self.var('params'),
            'point': [shape.x, shape.y],
            'resolution': args.resolution,
        })

        gws.log.debug(f'WMS_QUERY: START')
        gws.p(fa)

        fs = self.provider.find_features(fa)

        if fs is None:
            gws.log.debug('WMS_QUERY: NOT_PARSED')
            return []

        gws.log.debug(f'WMS_QUERY: FOUND {len(fs)}')
        return fs
