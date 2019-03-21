import gws
import gws.config.parser
import gws.common.format
import gws.common.search
import gws.common.template
import gws.auth.api
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
    maxAge: t.duration = '30d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache


class GridConfig(t.Config):
    """Grid configuration for caches and tiled data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    metaSize: int = 0  #: number of meta-tiles to fetch
    metaBuffer: int = 0  #: pixel buffer


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'  #: png 8-bit
    png24 = 'png24'  #: png 24-bit


class DisplayMode(t.Enum):
    """Layer display mode"""

    box = 'box'  #: display a layer as one big image (WMS-alike)
    tile = 'tile'  #: display a layer in a tile grid


class BaseConfig(t.WithTypeAndAccess):
    """Layer"""

    clientOptions: ClientOptions = {}  #: options for the layer display in the client
    description: t.Optional[t.TemplateConfig]  #: template for the layer description
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    featureFormat: t.Optional[t.FormatConfig]  #: feature formatting options
    legend: LegendConfig = {}  #: legend configuration
    meta: t.Optional[t.MetaConfig]  #: layer meta data
    opacity: float = 1  #: layer opacity
    search: gws.common.search.Config = {}  #: layer search configuration
    title: str  #: layer title
    uid: str = ''  #: layer unique id
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class ImageConfig(BaseConfig):
    cache: CacheConfig = {}  #: cache configuration
    display: DisplayMode = 'box'  #: layer display mode
    grid: GridConfig = {}  #: grid configuration
    imageFormat: ImageFormat = 'png8'  #: image format


class ImageTileConfig(ImageConfig):
    display: DisplayMode = 'tile'  #: layer display mode


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
        self.is_public = False

        self.has_cache = False

        self.layers = []

        self.map = None
        self.meta = None

        self.description_template: t.TemplateObject = None
        self.feature_format: t.FormatInterface = None

        self.title = None

        self.resolutions = []
        self.extent = []

        self.has_legend = False
        self.legend_url = None

        self.opacity = None
        self.client_options = None

    @property
    def has_search(self):
        return len(self.get_children('gws.ext.search.provider')) > 0

    def configure(self):
        super().configure()

        m = self.var('meta')
        title = self.var('title')
        if m:
            self.meta = m
            # title at the top level config preferred
            if title:
                self.meta.title = title
        else:
            self.meta = t.MetaData({
                'title': title
            })

        self.title = self.meta.title

        self.uid = self.var('uid') or gws.as_uid(self.title)

        self.map = self.get_closest('gws.common.map')
        if self.map:
            self.uid = self.map.uid + '.' + self.uid

        self.is_public = gws.auth.api.role('all').can_use(self)

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

        self.resolutions = gws.gis.zoom.resolutions_from_config(
            self.var('zoom'),
            self.map.resolutions)

        self.extent = self.var('extent') or self.map.extent

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
        return gws.compact({
            'meta': self.meta,
            'opacity': self.opacity,
            'options': self.client_options,
            'extent': self.extent if self.extent != self.map.extent else None,
            'resolutions': self.resolutions if self.resolutions != self.map.resolutions else None,
            'title': self.title,
            'uid': self.uid,
            'description': self.description,
        })

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
        if self.has_legend:
            return gws.ows.request.raw_get(self.legend_url).content

    def get_features(self, bbox, limit=0):
        return []

    def modify_features(self, operation, feature_params):
        pass

    def is_enabled_for_service(self, service):
        return False


class Image(Base):
    def __init__(self):
        super().__init__()

        self.display = ''
        self.image_format = ''

        self.cache: CacheConfig = None
        self.grid: GridConfig = None

        self.cache_uid = None
        self.grid_uid = None

    def configure(self):
        super().configure()

        self.image_format = self.var('imageFormat')
        self.display = self.var('display')

        self.cache = self.var('cache')
        self.has_cache = self.cache and self.cache.enabled

        self.grid = self.var('grid')

    def render_bbox(self, bbox, width, height, **client_params):
        uid = self.uid
        if not self.has_cache:
            uid += '_NOCACHE'
        return gws.gis.mpx.wms_request(uid, bbox, width, height, self.map.crs)

    def render_xyz(self, x, y, z):
        return gws.gis.mpx.wmts_request(
            self.uid,
            x, y, z,
            tile_matrix=self.grid_uid,
            tile_size=self.grid.tileSize)

    def is_enabled_for_service(self, service):
        return service == 'wms'

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

    def mapproxy_layer_config(self, mc, source_uid):

        mc.layer({
            'name': self.uid + '_NOCACHE',
            'sources': [source_uid]
        })

        res = [r for r in self.resolutions if r]
        if len(res) < 2:
            res = [res[0], res[0]]

        self.grid_uid = mc.grid(gws.compact({
            'origin': self.grid.origin,
            'tile_size': [self.grid.tileSize, self.grid.tileSize],
            'res': res,
            'srs': self.map.crs,
            'bbox': self.extent,
        }))

        meta_size = self.grid.metaSize or 4

        front_cache_config = {
            'sources': [source_uid],
            'grids': [self.grid_uid],
            'cache': {
                'type': 'file',
                'directory_layout': 'mp'
            },
            'meta_size': [meta_size, meta_size],
            'meta_buffer': self.grid.metaBuffer,
            'disable_storage': not self.has_cache,
            'minimize_meta_requests': True,
            'format': self.image_format,
        }

        self.cache_uid = mc.cache(front_cache_config)

        mc.layer({
            'name': self.uid,
            'sources': [self.cache_uid]
        })

    def mapproxy_back_cache_config(self, mc, url, grid_uid):
        source_uid = mc.source({
            'type': 'tile',
            'url': url,
            'grid': grid_uid,
            'concurrent_requests': self.var('maxRequests', default=0)
        })

        return mc.cache(gws.compact({
            'sources': [source_uid],
            'grids': [grid_uid],
            'cache': {
                'type': 'file',
                'directory_layout': 'mp'
            },
            'disable_storage': True,
            # 'meta_size': [1, 1],
            # 'meta_buffer': 0,
            # 'minimize_meta_requests': True,
        }))

    @property
    def props(self):
        p = super().props

        if self.display == 'tile':
            return gws.extend(p, {
                'type': 'tile',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetXyz/layerUid/' + self.uid + '/z/{z}/x/{x}/y/{y}/t.png',
                'tileSize': self.grid.tileSize,
            })

        if self.display == 'box':
            return gws.extend(p, {
                'type': 'box',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBbox/layerUid/' + self.uid,
            })


class ImageTile(Image):
    def configure(self):
        super().configure()

        # with meta=1 MP will request the same tile multiple times
        # meta=4 is more efficient, however, meta=1 yields the first tile faster
        # which is crucial when browsing non-cached low resoltions
        # so, let's use 1 as default, overridable in the config
        #
        # @TODO make MP cache network requests

        self.grid.metaSize = self.grid.metaSize or 1


class Vector(Base):
    @property
    def props(self):
        return gws.extend(super().props, {
            'dataModel': self.var('dataModel'),
            'editStyle': self.var('editStyle'),
            'loadingStrategy': self.var('loadingStrategy'),
            'style': self.var('style'),
            'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetFeatures/layerUid/' + self.uid,
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


def add_meta_from_source_layers(obj):
    if obj.var('meta'):
        return
    sl = getattr(obj, 'source_layers', [])
    if sl and len(sl) == 1:
        obj.meta = sl[0].meta
