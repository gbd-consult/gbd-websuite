"""QGIS WMS search."""

import shapely.geometry.point

import gws
import gws.common.search.provider
import gws.gis.source

import gws.types as t

from . import provider


class Config(gws.common.search.provider.Config):
    """Qgis/WMS automatic search provider"""

    path: t.FilePath  #: project path
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []

    def configure(self):
        super().configure()

        self.provider = provider.create_shared(self, self.config)
        self.source_layers = gws.gis.source.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
            queryable_only=True)

        # don't raise any errors here, because it would make parent configuration harder
        # see also layer/wms

    def can_run(self, args):
        return (
                self.source_layers
                and args.shapes
                and args.shapes[0].type == 'Point'
                and not args.keyword
        )

    def run(self, layer: t.LayerObject, args: t.SearchArguments) -> t.List[t.Feature]:
        qgis_crs = self.provider.supported_crs[0]

        shape = args.shapes[0]
        if args.crs != qgis_crs:
            shape = shape.transform(qgis_crs)

        geo = t.cast(shapely.geometry.point.Point, shape.geo)

        args = t.SearchArguments({
            'bbox': shape.bounds,
            'count': args.limit,
            'layers': [sl.name for sl in self.source_layers],
            'point': [geo.x, geo.y],
            'resolution': args.resolution,
        })

        gws.log.debug(f'QGIS_WMS_QUERY: START')
        gws.p(args)

        fs = self.provider.find_features(args)

        if fs is None:
            gws.log.debug('QGIS_WMS_QUERY: NOT_PARSED')
            return []

        gws.log.debug(f'QGIS_WMS_QUERY: FOUND {len(fs)}')
        return fs
