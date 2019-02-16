import gws
import gws.config
import gws.gis.feature
import gws.gis.proj
import gws.ows.wms
import gws.ows.util
import gws.gis.shape
import gws.common.search.provider
import gws.gis.source
import gws.tools.net
import gws.types as t


class Config(gws.common.search.provider.Config):
    """WMS search"""

    axis: str = ''  #: force axis orientation (axis=xy or axis=yx)
    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    maxRequests: int = 0  #: max concurrent requests to this source
    params: t.Optional[dict]  #: query string parameters
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: t.url  #: service url


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.service: gws.ows.wms.Service = None

        self.axis = ''
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''

    def configure(self):
        super().configure()

        self.axis = self.var('axis')

        self.url = self.var('url')
        self.service = gws.ows.util.shared_service('WMS', self, self.config)

        self.source_layers = gws.gis.source.filter_layers(
            self.service.layers,
            self.var('sourceLayers'),
            queryable_only=True)

        # don't raise any errors here, because it would make parent configuration harder
        # see also layer/wms, search/wms

    def can_run(self, args):
        return (
                self.source_layers
                and 'GetFeatureInfo' in self.service.operations
                and args.shapes
                and args.shapes[0].type == 'Point'
                and not args.keyword
        )

    def run(self, layer: t.LayerObject, args: t.SearchArgs) -> t.List[t.FeatureInterface]:
        shape = args.shapes[0]
        crs, shape = gws.ows.util.crs_and_shape(args.crs, self.service.supported_crs, shape)
        axis = gws.ows.util.axis_for(self.axis, 'WMS', self.service.version, crs)

        fa = t.FindFeaturesArgs({
            'axis': axis,
            'bbox': '',
            'count': args.limit,
            'crs': crs,
            'layers': [sl.name for sl in self.source_layers],
            'params': self.var('params'),
            'point': [shape.geo.x, shape.geo.y],
            'resolution': args.resolution,
        })

        gws.log.debug(f'WMS_QUERY: START')
        gws.p(fa)

        fs = self.service.find_features(fa)

        if fs is None:
            gws.log.debug('WMS_QUERY: NOT_PARSED')
            return []

        gws.log.debug(f'WMS_QUERY: FOUND {len(fs)}')
        return fs
