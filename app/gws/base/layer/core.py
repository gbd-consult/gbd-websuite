"""Base layer object.



Layer configuration protocol

    configure_bounds
    configure_grid
    configure_legend
    configure_metadata
    configure_models
    configure_resolutions
    configure_search
    configure_templates    




"""

import gws
import gws.base.legend
import gws.base.model
import gws.base.search
import gws.base.template
import gws.gis.bounds
import gws.gis.crs
import gws.gis.extent
import gws.gis.source
import gws.gis.zoom
import gws.lib.metadata
import gws.lib.style
import gws.lib.svg

import gws.types as t


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'
    """png 8-bit"""
    png24 = 'png24'
    """png 24-bit"""


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False
    """the layer is expanded in the list view"""
    listed: t.Optional[bool] = True
    """the layer is displayed in this list view"""
    selected: t.Optional[bool] = False
    """the layer is intially selected"""
    visible: t.Optional[bool] = True
    """the layer is intially visible"""
    unfolded: t.Optional[bool] = False
    """the layer is not listed, but its children are"""
    exclusive: t.Optional[bool] = False
    """only one of this layer's children is visible at a time"""


class CacheConfig(gws.Config):
    """Cache configuration"""

    enabled: bool = True
    maxAge: gws.Duration = '7d'
    """cache max. age"""
    maxLevel: int = 1
    """max. zoom level to cache"""
    requestBuffer: t.Optional[int]
    requestTiles: t.Optional[int]


class GridConfig(gws.Config):
    """Grid configuration for caches and tiled data"""

    crs: t.Optional[gws.CrsName]
    extent: t.Optional[gws.Extent]
    corner: t.Optional[gws.Corner]
    resolutions: t.Optional[t.List[float]]
    tileSize: t.Optional[int]


class EditConfig(gws.ConfigWithAccess):
    """Edit access for a layer"""

    pass


class SearchConfig(gws.Config):
    enabled: bool = False
    """search is enabled"""


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    models: t.Optional[t.List[gws.ext.config.model]]
    """data models"""
    cache: t.Optional[CacheConfig]
    """cache configuration"""
    clientOptions: ClientOptions = {}
    """options for the layer display in the client"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.box
    """layer display mode"""
    extent: t.Optional[gws.Extent]
    """layer extent"""
    extentBuffer: t.Optional[int]
    """extent buffer"""
    sourceGrid: t.Optional[GridConfig]
    """source grid"""
    grid: t.Optional[GridConfig]
    """target (client) grid"""
    imageFormat: ImageFormat = ImageFormat.png8
    """image format"""
    legend: t.Optional[gws.ext.config.legend]
    """legend configuration"""
    metadata: t.Optional[gws.Metadata]
    """layer metadata"""
    opacity: float = 1
    """layer opacity"""
    ows: bool = True  # layer is enabled for OWS services
    search: t.Optional[SearchConfig]
    """layer search configuration"""
    finders: t.Optional[t.List[gws.ext.config.finder]]
    """search prodivers"""
    templates: t.Optional[t.List[gws.ext.config.template]]
    """client templates"""
    title: str = ''
    """layer title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """layer resolutions and scales"""
    loadingStrategy: gws.LayerLoadingStrategy = gws.LayerLoadingStrategy.all
    """loading strategy for features"""
    cssSelector: t.Optional[str]
    """css selector for features"""


class CustomConfig(gws.ConfigWithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.gis.source.LayerFilter]
    """source layers this configuration applies to"""
    clientOptions: t.Optional[ClientOptions]  # options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]
    """layer data model"""
    display: t.Optional[gws.LayerDisplayMode]
    """layer display mode"""
    extent: t.Optional[gws.Extent]
    """layer extent"""
    extentBuffer: t.Optional[int]
    """extent buffer"""
    legend: gws.base.legend.Config = {}  # type:ignore
    """legend configuration"""
    metadata: t.Optional[gws.Metadata]
    """layer metadata"""
    opacity: t.Optional[float]
    """layer opacity"""
    ows: bool = True  # layer is enabled for OWS services
    # search: gws.base.search.finder.collection.Config = {}  # type:ignore
    """layer search configuration"""
    templates: t.Optional[t.List[gws.ext.config.template]]
    """client templates"""
    title: t.Optional[str]
    """layer title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """layer resolutions and scales"""


class GridProps(gws.Props):
    corner: str
    extent: gws.Extent
    resolutions: t.List[float]
    tileSize: int


class Props(gws.Props):
    model: t.Optional[gws.base.model.Props]
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    metadata: gws.lib.metadata.Props
    opacity: t.Optional[float]
    clientOptions: ClientOptions
    resolutions: t.Optional[t.List[float]]
    displayMode: str
    # style: t.Optional[gws.lib.style.Props]
    grid: GridProps
    title: str = ''
    type: str
    uid: str
    url: str = ''


_DEFAULT_STYLE = gws.Config(
    values={
        'fill': 'rgba(0,0,0,1)',
        'stroke': 'rgba(0,0,0,1)',
        'stroke-width': 1,
    }
)

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/layer_description.cx.html',
        subject='layer.description',
        access=gws.PUBLIC,
        uid='gws.base.layer.templates.layer_description',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_description.cx.html',
        subject='feature.description',
        access=gws.PUBLIC,
        uid='gws.base.layer.templates.feature_description',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_teaser.cx.html',
        subject='feature.teaser',
        access=gws.PUBLIC,
        uid='gws.base.layer.templates.feature_teaser',
    ),
]

_DEFAULT_TILE_SIZE = 256


class Object(gws.Node, gws.ILayer):
    clientOptions: ClientOptions

    canRenderBox = False
    canRenderXyz = False
    canRenderSvg = False

    supportsRasterServices = False
    supportsVectorServices = False

    parentBounds: gws.Bounds
    parentResolutions: t.List[float]

    def configure(self):
        self.parentBounds = self.var('_parentBounds')
        self.parentResolutions = self.var('_parentResolutions')

        self.bounds = self.parentBounds
        self.clientOptions = self.var('clientOptions')
        self.displayMode = self.var('display')
        self.loadingStrategy = self.var('loadingStrategy')
        self.imageFormat = self.var('imageFormat')
        self.opacity = self.var('opacity')
        self.resolutions = self.parentResolutions
        self.title = self.var('title')

        self.isSearchable = self.var('search.enabled')

        self.templates = []
        self.models = []
        self.finders = []

        self.metadata = gws.Metadata()
        self.legend = None

        self.layers = []

        self.sourceGrid = None
        self.grid = None

        p = self.var('grid')
        if p and p.crs and p.crs != self.parentBounds.crs:
            raise gws.Error(f'invalid target grid crs')

        self.cache = None
        if self.var('cache.enabled'):
            self.cache = gws.LayerCache(self.var('cache'))

    def configure_bounds(self):
        p = self.var('extent')
        if p:
            self.bounds = gws.Bounds(
                crs=self.parentBounds.crs,
                extent=gws.gis.extent.from_list(p))
            return True

    def configure_grid(self):
        p = self.var('grid')
        if p:
            self.grid = gws.TileGrid(
                corner=p.corner or gws.Corner.nw,
                tileSize=p.tileSize or _DEFAULT_TILE_SIZE,
                bounds=gws.Bounds(crs=self.parentBounds.crs, extent=p.extent),
                resolutions=p.resolutions)
            return True

    def configure_legend(self):
        p = self.var('legend')
        if p and not p.enabled:
            return True
        if p and p.enabled and p.type:
            self.legend = self.create_child(gws.ext.object.legend, p)
            return True

    def configure_metadata(self):
        p = self.var('metadata')
        if p:
            self.metadata = gws.lib.metadata.from_config(p)
            return True

    def configure_resolutions(self):
        p = self.var('zoom')
        if p:
            self.resolutions = gws.gis.zoom.resolutions_from_config(p, self.parentResolutions)
            if not self.resolutions:
                raise gws.Error(f'self {self.uid!r}: no resolutions, config={p!r}, parent={self.parentResolutions!r}')
            return True

    def configure_models(self):
        p = self.var('models')
        if p:
            self.models = self.create_children(gws.ext.object.model, p)
            return True

    def configure_search(self):
        if not self.isSearchable:
            return True
        p = self.var('finders')
        if p:
            self.finders = self.create_children(gws.ext.object.finder, p)
            return True

    def configure_templates(self):
        self.templates = self.create_children(gws.ext.object.template, self.var('templates'))
        for cfg in _DEFAULT_TEMPLATES:
            self.templates.append(self.root.create_shared(gws.ext.object.template, cfg))
        return True

    ##

    def configure_group(self, layer_configs):
        has_resolutions = self.configure_resolutions()
        has_bounds = self.configure_bounds()

        ls = []

        for cfg in layer_configs:
            cfg = gws.merge(
                cfg,
                _parentBounds=self.bounds,
                _parentResolutions=self.resolutions,
            )
            ls.append(self.create_child(gws.ext.object.layer, cfg))

        self.layers = gws.compact(ls)
        if not self.layers:
            raise gws.Error(f'group {self.uid!r} is empty')

        if not has_resolutions:
            res = set()
            for la in self.layers:
                res.update(la.resolutions)
            self.resolutions = sorted(res)

        if not has_bounds:
            self.bounds = gws.gis.bounds.union([la.bounds for la in self.layers])

        if not self.configure_legend():
            layers_uids = [la.uid for la in self.layers if la.legend]
            if layers_uids:
                self.legend = self.create_child(
                    gws.ext.object.legend,
                    gws.Config(type='combined', layerUids=layers_uids))

        self.canRenderBox = any(la.canRenderBox for la in self.layers)
        self.canRenderXyz = any(la.canRenderXyz for la in self.layers)
        self.canRenderSvg = any(la.canRenderSvg for la in self.layers)

        self.supportsRasterServices = any(la.supportsRasterServices for la in self.layers)
        self.supportsVectorServices = any(la.supportsVectorServices for la in self.layers)

    ##

    def ancestors(self):
        ls = []
        p = self.parent
        while isinstance(p, Object):
            ls.append(p)
            p = p.parent
        return ls

    def descendants(self):
        ls = []
        for la in self.layers:
            ls.append(la)
            ls.extend(la.descendants())
        return ls

    _url_path_suffix = '/gws.png'

    def url_path(self, kind):
        # layer urls, handled by the map action (base/map/action.py)
        if kind == 'box':
            return gws.action_url_path('mapGetBox', layerUid=self.uid) + self._url_path_suffix
        if kind == 'tile':
            return gws.action_url_path('mapGetXYZ', layerUid=self.uid) + '/z/{z}/x/{x}/y/{y}' + self._url_path_suffix
        if kind == 'legend':
            return gws.action_url_path('mapGetLegend', layerUid=self.uid) + self._url_path_suffix
        if kind == 'features':
            return gws.action_url_path('mapGetFeatures', layerUid=self.uid)

    def props(self, user):
        p = gws.Data(
            clientOptions=self.clientOptions,
            displayMode=self.displayMode,
            extent=self.bounds.extent,
            layers=self.layers,
            loadingStrategy=self.loadingStrategy,
            metadata=gws.lib.metadata.props(self.metadata),
            opacity=self.opacity,
            resolutions=sorted(self.resolutions, reverse=True),
            title=self.title,
            uid=self.uid,
        )

        if self.grid:
            p.grid = GridProps(
                corner=self.grid.corner,
                extent=self.grid.bounds.extent,
                resolutions=sorted(self.grid.resolutions, reverse=True),
                tileSize=self.grid.tileSize,
            )

        if self.displayMode == gws.LayerDisplayMode.tile:
            p.type = 'tile'
            p.url = self.url_path('tile')

        if self.displayMode == gws.LayerDisplayMode.box:
            p.type = 'box'
            p.url = self.url_path('box')

        return p

    def render_legend(self, args=None) -> t.Optional[gws.LegendRenderOutput]:

        if not self.legend:
            return None

        def _get():
            out = self.legend.render()
            return out

        if not args:
            return gws.get_server_global('legend_' + self.uid, _get)

        return self.legend.render(args)

    def mapproxy_config(self, mc):
        pass
