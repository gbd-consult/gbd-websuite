import gws
import gws.config
import gws.gis.feature
import gws.gis.proj
import gws.qgis
import gws.gis.shape
import gws.common.search.provider
import gws.tools.net
import gws.types as t


class Config(gws.common.search.provider.Config):
    """Qgis/WMS automatic search provider"""

    path: t.filepath  #: project path
    layers: t.List[str]  #: layers to query


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()
        self.service: gws.qgis.Service = None
        self.crs = ''

    def configure(self):
        super().configure()
        self.service = gws.qgis.shared_service(self, self.config)

    def can_run(self, args):
        return (
            args.shapes
            and args.shapes[0].type == 'Point'
            and not args.keyword
        )

    def run(self, layer: t.LayerObject, args: t.SearchArgs) -> t.List[t.FeatureInterface]:
        qgis_crs = self.service.supported_crs[0]

        shape = args.shapes[0]
        if args.crs != qgis_crs:
            shape = shape.transform(qgis_crs)

        fa = t.FindFeaturesArgs({
            'bbox': shape.bounds,
            'count': args.limit,
            'layers': self.var('layers'),
            'point': [shape.geo.x, shape.geo.y],
            'resolution': args.resolution,
        })

        gws.log.debug(f'QGIS_WMS_QUERY: START')
        gws.p(fa, 2)

        fs = self.service.find_features(fa)

        if fs is None:
            gws.log.debug('QGIS_WMS_QUERY: NOT_PARSED')
            return []

        gws.log.debug(f'QGIS_WMS_QUERY: FOUND {len(fs)}')
        return fs
