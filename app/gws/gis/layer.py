import gws
import gws.config.parser
import gws.common.format
import gws.common.search
import gws.common.template
import gws.gis.feature
import gws.gis.mpx as mpx
import gws.gis.proj
import gws.gis.source
import gws.gis.zoom
import gws.ows.request
import gws.types as t

import gws.tools.net


class ClientOptions(t.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False  #: the layer is expanded in the list view
    listed: t.Optional[bool] = True  #: the layer is displayed in this list view
    selected: t.Optional[bool] = False  #: the layer is intially selected
    visible: t.Optional[bool] = True  #: the layer is intially visible
    unfolded: t.Optional[bool] = False  #: the layer is not listed, but its children are
    exclusive: t.Optional[bool] = False  #: only one of this layer's children is visible at a time


class EditConfig(t.Config):
    access: t.Access


class LegendConfig(t.Config):
    enabled: bool = True
    url: t.Optional[t.url]


class CacheConfig(t.Config):
    """Cache configuration"""

    enabled: bool = False  #: cache is enabled
    maxAge: t.duration = '1d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache
    options: dict = {}  #: additional MapProxy cache options


class GridConfig(t.Config):
    """Grid configuration for caches and tiled data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    metaSize: int = 4  #: number of meta-tiles to fetch
    metaBuffer: int = 200  #: pixel buffer
    options: dict = {}  #: additional MapProxy grid options


class BaseConfig(t.WithTypeAndAccess):
    """Layer"""

    cache: CacheConfig = {}  #: cache configuration
    clientOptions: ClientOptions = {}  #: options for the layer display in the client
    description: t.Optional[t.TemplateConfig]  #: template for the layer description
    display: str = ''  #: layer display mode ('box', 'tile', 'client')
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    featureFormat: t.Optional[t.FormatConfig]  #: feature formatting options
    grid: GridConfig = {}  #: grid configuration
    legend: LegendConfig = {}  #: legend configuration
    meta: t.MetaConfig = {}  #: layer meta data
    opacity: float = 1  #: layer opacity
    search: gws.common.search.Config = {}  #: layer search configuration
    title: str  #: layer title
    uid: str = ''  #: layer unique id
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class ProxiedConfig(BaseConfig):
    pass


class VectorConfig(BaseConfig):
    editStyle: t.Optional[t.StyleProps]  #: style for features being edited
    style: t.Optional[t.StyleProps]  #: style for features
    dataModel: t.Optional[t.List[t.AttributeConfig]]
    loadingStrategy: str = 'all'  #: loading strategy for features ('all', 'bbox')


class Props(t.Data):
    dataModel: t.Optional[t.List[t.AttributeConfig]]
    description: str = ''
    editAccess: t.Optional[t.List[str]]
    editStyle: t.Optional[t.StyleProps]
    extent: t.Optional[t.Extent]
    geometryType: str = ''
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    meta: t.MetaData
    opacity: t.Optional[float]
    options: ClientOptions
    resolutions: t.Optional[t.List[float]]
    style: t.Optional[t.StyleProps]
    tileSize: int = 0
    title: str = ''
    type: str
    uid: str
    url: str = ''


class Base(gws.PublicObject, t.LayerObject):
    def __init__(self):
        super().__init__()

        self.display = ''

        self.cache: CacheConfig = None
        self.has_cache = False
        self.cache_uid = None
        self.grid: GridConfig = None

        self.map = None
        self.meta = None

        self.description_template = None
        self.feature_format = None

        self.template = None
        self.title = None

        self.resolutions = []
        self.extent = []

        self.has_legend = False
        self.legend_url = None

        self.opacity = None
        self.client_options = None

    def configure(self):
        super().configure()

        self.meta = self.var('meta')
        # title at the top level config preferred
        if self.var('title'):
            self.meta.title = self.var('title')
        self.title = self.meta.title

        self.uid = self.var('uid') or gws.as_uid(self.title)

        self.map = self.get_closest('gws.common.map')
        if self.map:
            self.uid = self.map.uid + '.' + self.uid

        self.display = self.var('display')

        self.cache = self.var('cache')
        self.cache_uid = gws.as_uid(self.uid)
        self.has_cache = self.cache and self.cache.enabled

        self.grid = self.var('grid')

        p = self.var('legend')
        self.legend_url = p.url
        self.has_legend = p.enabled and p.url

        self.opacity = self.var('opacity')
        self.client_options = self.var('clientOptions')

        p = self.var('description')
        if p:
            self.description_template = self.create_object('gws.ext.template', p)
        else:
            self.description_template = self.create_shared_object(
                'gws.ext.template',
                'default_layer_description',
                gws.common.template.builtin_config('layer_description')
            )

        p = self.var('featureFormat')
        if p:
            self.feature_format = self.create_object('gws.common.format', p)
        else:
            self.feature_format = self.create_shared_object(
                'gws.common.format',
                'default_feature_description',
                gws.common.template.builtin_config('feature_format')
            )

        self.resolutions = gws.gis.zoom.effective_resolutions(
            self.var('zoom'),
            self.map.resolutions)

        self.extent = self.var('extent', parent=True)

        p = self.var('search')
        if p.enabled and p.providers:
            for cfg in p.providers:
                self.add_child('gws.ext.search.provider', cfg)

    def edit_access(self, user):
        # @TODO granular edit access

        if user.can_use(self.var('edit'), parent=self):
            return ['all']

    @property
    def props(self):

        p = gws.compact({
            'meta': self.meta,
            'opacity': self.opacity,
            'options': self.client_options,
            # @TODO: dont write those if equal to parent
            'extent': self.extent,
            'resolutions': self.resolutions,
            'title': self.title,
            'uid': self.uid,
            'description': self.description,
        })

        if self.display == 'tile':
            p = gws.extend(p, {
                'type': 'tile',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetXyz/layerUid/' + self.uid + '/z/{z}/x/{x}/y/{y}/t.png',
                'tileSize': self.grid.tileSize,
            })

        if self.display == 'box':
            p = gws.extend(p, {
                'type': 'box',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBbox/layerUid/' + self.uid,
            })

        return p

    @property
    def description(self):
        ctx = {
            'layer': self,
        }
        return self.description_template.render(ctx).content

    def props_for(self, user):
        p = super().props_for(user)
        if p:
            p['editAccess'] = self.edit_access(user)
        return p

    def mapproxy_config(self, mc):
        pass

    def render_bbox(self, bbox, width, height, **client_params):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, bbox, dpi, scale, rotation, style):
        return None

    def render_legend(self):
        if self.legend_url:
            return gws.ows.request.raw_get(self.legend_url).content

    def get_features(self, bbox, limit=0):
        return []

    def modify_features(self, operation, feature_params):
        pass


class Proxied(Base):
    def render_bbox(self, bbox, width, height, **client_params):
        cache_uid = self.cache_uid
        if not self.has_cache:
            cache_uid += '_NOCACHE'
        return gws.gis.mpx.wms_request(cache_uid, bbox, width, height, self.map.crs)

    def render_xyz(self, x, y, z):
        return gws.gis.mpx.wmts_request(
            self.cache_uid,
            x, y, z,
            tile_matrix='grid_' + self.cache_uid + '_FRONT',
            tile_size=self.grid.tileSize)

    """
        Mapproxy config is done in two steps
        
        1. first, configure the source. For box layers, this is a normal WMS source. 
        For tiled layers, we use the 'double cache' technique, see
    
        https://mapproxy.org/docs/nightly/configuration_examples.html#create-wms-from-existing-tile-server
        https://mapproxy.org/docs/1.11.0/configuration_examples.html#reprojecting-tiles
    
        Basically, the source is wrapped in a no-store BACK cache, which is then given to the front mpx layer
        
        2. then, configure the layer. Create the FRONT cache, which is store or no-store, depending on the cache setting.
        Also, configure the _NOCACHE variant for the layer, which skips the DST cache
    """

    def mapproxy_layer_config(self, mc, source):

        mc.layer(self.cache_uid + '_NOCACHE', {
            'sources': [source]
        })

        res = [r for r in self.resolutions if r]
        if len(res) < 2:
            res = [res[0], res[0]]

        front_grid_config = gws.compact({
            'origin': self.grid.origin,
            'tile_size': [self.grid.tileSize, self.grid.tileSize],
            'res': res,
            'srs': self.map.crs,
            'bbox': self.extent,
        })

        front_grid = mc.grid(self.cache_uid + '_FRONT', gws.extend(front_grid_config, self.grid.options))

        front_cache_options = {
            'type': 'file',
            'directory_layout': 'mp'
        }

        front_cache_config = {
            'sources': [source],
            'grids': [front_grid],
            'cache': gws.extend(front_cache_options, self.cache.options),
            'meta_size': [self.grid.metaSize, self.grid.metaSize],
            'meta_buffer': self.grid.metaBuffer,
            'disable_storage': not self.has_cache,
            'minimize_meta_requests': True,
        }

        front_cache = mc.cache(self.cache_uid + '_FRONT', front_cache_config)

        mc.layer(self.cache_uid, {
            'sources': [front_cache]
        })

    def mapproxy_back_cache_config(self, mc, url, src_grid_config):
        grid = mc.grid(self.cache_uid + '_BACK', src_grid_config)

        source = mc.source(self.cache_uid, {
            'type': 'tile',
            'url': url,
            'grid': grid,
            'concurrent_requests': self.var('maxRequests', default=0)
        })

        src_cache_options = {
            'type': 'file',
            'directory_layout': 'mp'
        }

        src_cache_config = gws.compact({
            'sources': [source],
            'grids': [grid],
            'cache': src_cache_options,
            'disable_storage': True,
            'meta_size': [1, 1],
            'meta_buffer': 0,
            'minimize_meta_requests': True,
        })

        return mc.cache(self.cache_uid + '_BACK', src_cache_config)


class ProxiedTile(Proxied):

    def configure(self):
        super().configure()

        self.display = self.var('display', default='tile')

        # force no meta for tiled layers, otherwise MP keeps requested the same tile multiple times

        self.grid = t.Config(gws.extend(self.grid, {
            'metaSize': 1,
            'metaBuffer': 0,
        }))


class Vector(Base):
    @property
    def props(self):
        return gws.extend(super().props, {
            'dataModel': self.var('dataModel'),
            'editStyle': self.var('editStyle'),
            'loadingStrategy': self.var('loadingStrategy'),
            'style': self.var('style'),
        })

    def render_svg(self, bbox, dpi, scale, rotation, style):
        features = self.get_features(bbox)
        for f in features:
            f.set_default_style(style)
        return [f.to_svg(bbox, dpi, scale, rotation) for f in features]


def add_layers_to_object(obj, layer_configs):
    ls = []
    for p in layer_configs:
        try:
            ls.append(obj.add_child('gws.ext.layer', p))
        except Exception as e:
            uid = gws.get(p, 'uid')
            gws.log.error(f'FAILED LAYER: map={obj.uid!r} layer={uid!r} error={e!r}')
            gws.log.exception()
    return ls
