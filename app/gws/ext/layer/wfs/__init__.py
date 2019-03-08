import gws
import gws.gis.layer
import gws.gis.proj
import gws.gis.shape
import gws.gis.source
import gws.ows.util
import gws.ows.wfs
import gws.common.search.provider

import gws.types as t


class WfsServiceConfig(t.Config):
    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[t.crsref]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: t.url  #: service url


class Config(gws.gis.layer.VectorConfig, WfsServiceConfig):
    """WFS layer"""
    pass


class Object(gws.gis.layer.Vector):
    def __init__(self):
        super().__init__()

        self.invert_axis = []
        self.service: gws.ows.wfs.Service = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.geometry_type = ''
        self.url = ''

    def configure(self):
        super().configure()

        _configure_wfs(self)

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        self.source_layers = [self.source_layers[0]]
        self._add_default_search()

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.service.meta,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'vector',
            ##'geometryType': self.geometry_type.upper(),
        })

    def get_features(self, bbox, limit=0):
        crs = self.map.crs
        service_crs = gws.ows.util.best_crs(crs, self.service.supported_crs)
        if service_crs != crs:
            bbox = gws.gis.proj.transform_bbox(bbox, crs, service_crs)
        fs = _find_features(self, bbox, service_crs, limit)
        if service_crs != crs:
            fs = [f.transform(crs) for f in fs]
        return fs

    def _add_default_search(self):
        p = self.var('search')
        if not p.enabled or p.providers:
            return

        cfg = {
            'type': 'wfs'
        }

        cfg_keys = [
            'capsCacheMaxAge',
            'invertAxis',
            'maxRequests',
            'sourceLayers',
            'url',
        ]

        for key in cfg_keys:
            cfg[key] = self.var(key)

        self.add_child('gws.ext.search.provider', t.Config(gws.compact(cfg)))


class SearchConfig(gws.common.search.provider.Config, WfsServiceConfig):
    pass


class SearchObject(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.invert_axis = []
        self.service: gws.ows.wfs.Service = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''

    def configure(self):
        super().configure()
        _configure_wfs(self)

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
        fs = _find_features(self, shape.bounds, crs, args.limit)

        # @TODO excluding geometryless features for now
        fs = [f for f in fs if f.shape and f.shape.geo.intersects(shape.geo)]
        gws.log.debug(f'WFS_QUERY: AFTER FILTER: {len(fs)}')

        return fs


def _configure_wfs(obj):
    obj.url = obj.var('url')

    obj.service = gws.ows.util.shared_service('WFS', obj, obj.config)
    obj.invert_axis = obj.var('invertAxis')
    obj.source_layers = gws.gis.source.filter_layers(
        obj.service.layers,
        obj.var('sourceLayers'),
        queryable_only=True)


def _find_features(obj, bbox, crs, limit) -> t.List[t.FeatureInterface]:
    axis = gws.ows.util.best_axis(crs, obj.invert_axis, 'WFS', obj.service.version)

    fa = t.FindFeaturesArgs({
        'axis': axis,
        'bbox': bbox,
        'count': limit,
        'crs': crs,
        'layers': [sl.name for sl in obj.source_layers],
        'point': '',
    })

    gws.log.debug(f'WFS_QUERY: START')
    gws.p(fa, d=2)

    fs = obj.service.find_features(fa)

    if fs is None:
        gws.log.debug('WFS_QUERY: NOT_PARSED')
        return []

    gws.log.debug(f'WFS_QUERY: FOUND {len(fs)}')
    return fs
