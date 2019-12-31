import gws

import gws.common.auth
import gws.common.format
import gws.common.metadata
import gws.common.model
import gws.common.ows.provider
import gws.common.search
import gws.common.style
import gws.common.template
import gws.config.parser
import gws.gis.extent
import gws.gis.feature
import gws.gis.mpx as mpx
import gws.gis.ows
import gws.gis.proj
import gws.gis.shape
import gws.gis.source
import gws.gis.svg
import gws.gis.zoom
import gws.tools.misc
import gws.tools.net

import gws.types as t


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
    url: t.Optional[t.Url]


class CacheConfig(t.Config):
    """Cache configuration"""

    enabled: bool = False  #: cache is enabled
    maxAge: t.Duration = '7d'  #: cache max. age
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
    client = 'client'  #: draw a layer in the client


class FlattenConfig(t.Config):
    """Flatten the layer hierarchy."""

    level: int  #: flatten level
    useGroups: bool = False  #: use group names (true) or image layer names (false)


class OwsConfig(t.Config):
    name: str = ''  #: layer name for ows services
    servicesEnabled: t.Optional[t.List[str]]  #: services enabled for this layer
    servicesDisabled: t.Optional[t.List[str]]  #: services disabled for this layer


class BaseConfig(t.WithTypeAndAccess):
    """Layer"""

    clientOptions: ClientOptions = {}  #: options for the layer display in the client
    dataModel: t.Optional[gws.common.model.Config]  #: layer data model
    description: t.Optional[t.ext.template.Config]  #: template for the layer description
    display: DisplayMode = 'box'  #: layer display mode
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: feature formatting options
    legend: LegendConfig = {}  #: legend configuration
    meta: t.Optional[gws.common.metadata.Config]  #: layer meta data
    opacity: float = 1  #: layer opacity
    ows: t.Optional[OwsConfig]  #: OWS services options
    search: gws.common.search.Config = {}  #: layer search configuration
    title: str = ''  #: layer title
    uid: str = ''  #: layer unique id
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class ImageConfig(BaseConfig):
    cache: CacheConfig = {}  #: cache configuration
    grid: GridConfig = {}  #: grid configuration
    imageFormat: ImageFormat = 'png8'  #: image format


class ImageTileConfig(ImageConfig):
    display: DisplayMode = 'tile'  #: layer display mode


class VectorConfig(BaseConfig):
    display: DisplayMode = 'client'  #: layer display mode
    editDataModel: t.Optional[gws.common.model.Config]  #: data model for input data
    editStyle: t.Optional[gws.common.style.Config]  #: style for features being edited
    loadingStrategy: str = 'all'  #: loading strategy for features ('all', 'bbox')
    style: t.Optional[gws.common.style.Config]  #: style for features


class Props(t.Data):
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


#:export ILayer
class Base(gws.Object, t.ILayer):
    def __init__(self):
        super().__init__()

        self.display = ''
        self.image_format = ''

        self.is_public = False

        self.has_cache = False

        #:noexport
        self.cache: CacheConfig = None
        #:noexport
        self.grid: GridConfig = None

        self.cache_uid = None
        self.grid_uid = None

        self.layers = []

        self.map: t.IMap = None
        self.meta: t.MetaData = None

        self.description_template: t.ITemplate = None
        self.feature_format: t.IFormat = None
        self.data_model: t.IModel = None

        self.title = ''

        self.resolutions = []
        self.extent = []
        self.crs = ''

        self.own_crs = ''

        self.has_legend = False
        self.legend_url = ''

        self.opacity = 1
        self.client_options = t

        self.services = []
        self.geometry_type = None

        self.style: t.IStyle = None
        self.edit_style: t.IStyle = None
        self.edit_data_model: t.IModel = None

        self.can_render_bbox = False
        self.can_render_xyz = False
        self.can_render_svg = False

        self.ows_name = ''

    @property
    def has_search(self) -> bool:
        return len(self.get_children('gws.ext.search.provider')) > 0

    @property
    def own_bounds(self) -> t.Bounds:
        pass

    def use_meta(self, meta):
        title = self.var('title')
        if meta:
            # title at the top level config preferred
            if title:
                meta.title = title
            self.meta = meta
        else:
            self.meta = t.MetaData({
                'title': title
            })
        self.title = self.meta.title

        uid = self.var('uid') or gws.as_uid(self.title) or 'layer'
        self.map = self.get_closest('gws.common.map')
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        self.use_meta(gws.common.metadata.read(self.var('meta')))
        self.is_public = gws.common.auth.role('all').can_use(self)

        self.ows_name = self.var('ows.name') or self.uid.split('.')[-1]

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

        self.crs = self.var('crs') or self.map.crs

        p = self.var('search')
        if p.enabled and p.providers:
            for cfg in p.providers:
                self.add_child('gws.ext.search.provider', cfg)

        p = self.var('dataModel')
        if p:
            self.data_model = self.add_child('gws.common.model', p)

        self.image_format = self.var('imageFormat')
        self.display = self.var('display')

        self.ows_services_enabled = set(self.var('ows.servicesEnabled', default=[]))
        self.ows_services_disabled = set(self.var('ows.servicesDisabled', default=[]))

        p = self.var('editDataModel')
        if p:
            self.edit_data_model = self.add_child('gws.common.model', p)

        p = self.var('style')
        if p:
            self.style = gws.common.style.from_config(p)

        p = self.var('editStyle')
        if p:
            self.edit_style = gws.common.style.from_config(p)

        self.cache = self.var('cache')
        self.has_cache = self.cache and self.cache.enabled

        self.grid = self.var('grid')

    def edit_access(self, user):
        # @TODO granular edit access

        if user.can_use(self.var('edit'), parent=self):
            return ['all']

    def edit_operation(self, operation: str, feature_props: t.List[t.FeatureProps]) -> t.List[t.IFeature]:
        pass

    @property
    def props(self):
        return Props({
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
    def description(self) -> str:
        ctx = {
            'layer': self,
        }
        return self.description_template.render(ctx).content

    def props_for(self, user):
        p = super().props_for(user)
        if p:
            p.editAccess = self.edit_access(user)
        return p

    def mapproxy_config(self, mc):
        pass

    def render_bbox(self, rv: t.RenderView, client_params=None):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, rv: t.RenderView, style=None):
        return None

    def render_legend(self):
        if not self.has_legend:
            return
        if self.legend_url.startswith('/'):
            with open(self.legend_url, 'rb') as fp:
                return fp.read()
        return gws.gis.ows.request.raw_get(self.legend_url).content

    def get_features(self, bounds: t.Bounds, limit: int = 0) -> t.List[t.IFeature]:
        return []

    def ows_enabled(self, service):
        if service.name in self.ows_services_disabled:
            return False
        if self.ows_services_enabled:
            return service.name in self.ows_services_enabled
        return True


class Image(Base):
    def __init__(self):
        super().__init__()

        self.can_render_bbox = True
        self.can_render_xyz = True

    def render_bbox(self, rv, client_params=None):
        uid = self.uid
        if not self.has_cache:
            uid += '_NOCACHE'
        return gws.gis.mpx.wms_request(uid, rv.bbox, rv.size_px[0], rv.size_px[1], self.map.crs)

    def render_xyz(self, x, y, z):
        return gws.gis.mpx.wmts_request(
            self.uid,
            x, y, z,
            tile_matrix=self.grid_uid,
            tile_size=self.grid.tileSize)

    def ows_enabled(self, service):
        return super().ows_enabled(service) and service.type == 'wms'

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
            'format': self.image_format,
        }))

    @property
    def props(self):
        if self.display == 'tile':
            return super().props.extend({
                'type': 'tile',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetXyz/layerUid/' + self.uid + '/z/{z}/x/{x}/y/{y}/t.png',
                'tileSize': self.grid.tileSize,
            })

        if self.display == 'box':
            return super().props.extend({
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


#:export IVectorLayer
class Vector(Base, t.IVectorLayer):
    def __init__(self):
        super().__init__()

        self.can_render_bbox = True
        self.can_render_svg = True

    @property
    def props(self):
        if self.display == 'box':
            return super().props.extend({
                'type': 'box',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBbox/layerUid/' + self.uid,
            })

        return super().props.extend({
            'type': 'vector',
            'loadingStrategy': self.var('loadingStrategy'),
            'style': self.style,
            'editStyle': self.edit_style,
            'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetFeatures/layerUid/' + self.uid,
        })

    def connect_feature(self, feature: t.IFeature) -> t.IFeature:
        feature.layer = self
        return feature

    def render_bbox(self, rv, client_params=None):
        elements = self.render_svg(rv)
        return gws.gis.svg.to_png(elements, size=rv.size_px)

    def render_svg(self, rv, style=None):
        features = self.get_features(rv.bounds)
        for f in features:
            f.convert(target_crs=rv.bounds.crs)
        return [f.to_svg(rv, style or self.style) for f in features]

    def ows_enabled(self, service):
        return super().ows_enabled(service) and service.type == 'wfs'


def add_layers_to_object(obj, layer_configs):
    ls = []
    for p in layer_configs:
        try:
            ls.append(obj.add_child('gws.ext.layer', p))
        except Exception as e:
            uid = gws.get(p, 'uid')
            gws.log.error(f'FAILED LAYER: map={obj.uid!r} layer={uid!r} error={e!r}')
            gws.log.exception()
            # @TODO: should be an option
            raise
    return ls


def meta_from_source_layers(layer):
    sl = getattr(layer, 'source_layers', [])
    if sl and len(sl) == 1:
        return sl[0].meta
