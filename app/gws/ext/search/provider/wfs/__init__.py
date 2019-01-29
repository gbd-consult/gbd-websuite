import gws
import gws.common.search.provider
import gws.ows.wfs
import gws.types as t
import gws.ows.util
import gws.gis.shape


class Config(gws.common.search.provider.Config):
    """WFS search"""

    axis: str = ''  #: force axis orientation (axis=xy or axis=yx)
    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    crs: t.Optional[t.crsref]  #: always use this crs
    layers: t.Optional[t.List[str]]  #: feature types to search for
    params: t.Optional[dict]  #: additional query parameters
    url: t.url  #: service url


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()
        self.service: gws.ows.wfs.Service = None
        self.axis = ''
        self.crs = ''
        self.layers = []
        self.url = ''

    def configure(self):
        super().configure()

        self.axis = self.var('axis')
        self.url = self.var('url')
        self.service = gws.ows.util.shared_service('WFS', self, self.config)
        self.layers = self.var('layers')
        if not self.layers:
            self.layers = [la.name for la in self.service.layers]

    def can_run(self, args):
        return (
            'GetFeature' in self.service.operations
            and args.shapes
            and not args.keyword
        )

    def run(self, layer: t.LayerObject, args: t.SearchArgs) -> t.List[t.FeatureInterface]:
        shape = gws.gis.shape.union(args.shapes)
        if shape.type == 'Point':
            shape = shape.tolerance_buffer(args.get('tolerance'))

        crs, shape = gws.ows.util.crs_and_shape(args.crs, self.service.supported_crs, shape)
        axis = gws.ows.util.axis_for(self.axis, 'WFS', self.service.version, crs)

        fa = t.FindFeaturesArgs({
            'axis': axis,
            'bbox': shape.bounds,
            'count': args.limit,
            'crs': crs,
            'layers': self.layers,
            'params': self.var('params'),
            'point': '',
            'resolution': args.resolution,
        })

        gws.log.debug(f'WFS_QUERY: START')
        gws.p(fa, 2)

        fs = self.service.find_features(fa)

        if fs is None:
            gws.log.debug('WFS_QUERY: NOT_PARSED')
            return []

        gws.log.debug(f'WFS_QUERY: FOUND {len(fs)}')

        # @TODO excluding geometryless features for now

        fs = [f for f in fs if f.shape and f.shape.geo.intersects(shape.geo)]

        gws.log.debug(f'WFS_QUERY: AFTER FILTER: {len(fs)}')

        return fs
