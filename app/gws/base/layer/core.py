"""Base layer object."""

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


class LayerOptions(gws.Data):
    search: t.Optional[bool] = True
    """layer is searchable"""
    legend: t.Optional[bool] = True
    """layer has a legend"""
    cache: t.Optional[bool] = False
    """layer is cached"""
    ows: t.Optional[bool] = True
    """layer is enabled for OWS services"""


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
    origin: t.Optional[gws.Origin]
    resolutions: t.Optional[list[float]]
    tileSize: t.Optional[int]


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    cache: t.Optional[CacheConfig]
    """cache configuration"""
    clientOptions: ClientOptions = {}
    """options for the layer display in the client"""
    cssSelector: str = ''
    """css selector for feature layers"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.box
    """layer display mode"""
    extent: t.Optional[gws.Extent]
    """layer extent"""
    extentBuffer: t.Optional[int]
    """extent buffer"""
    finders: t.Optional[list[gws.ext.config.finder]]
    """search prodivers"""
    grid: t.Optional[GridConfig]
    """client grid"""
    imageFormat: gws.ImageFormat = gws.ImageFormat.png8
    """image format"""
    legend: t.Optional[gws.ext.config.legend]
    """legend configuration"""
    loadingStrategy: gws.FeatureLoadingStrategy = gws.FeatureLoadingStrategy.all
    """feature loading strategy"""
    metadata: t.Optional[gws.Metadata]
    """layer metadata"""
    models: t.Optional[list[gws.ext.config.model]]
    """data models"""
    opacity: float = 1
    """layer opacity"""
    options: LayerOptions = {}
    """options for the layer"""
    templates: t.Optional[list[gws.ext.config.template]]
    title: str = ''
    zoom: t.Optional[gws.gis.zoom.Config]


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
    templates: t.Optional[list[gws.ext.config.template]]
    """client templates"""
    title: t.Optional[str]
    """layer title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """layer resolutions and scales"""


class GridProps(gws.Props):
    origin: str
    extent: gws.Extent
    resolutions: list[float]
    tileSize: int


class Props(gws.Props):
    clientOptions: ClientOptions
    cssSelector: str
    displayMode: str
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    grid: GridProps
    layers: t.Optional[list['Props']]
    loadingStrategy: gws.FeatureLoadingStrategy
    metadata: gws.lib.metadata.Props
    model: t.Optional[gws.base.model.Props]
    opacity: t.Optional[float]
    resolutions: t.Optional[list[float]]
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
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_title.cx.html',
        subject='feature.title',
        access=gws.PUBLIC,
        uid='gws.base.layer.templates.feature_title',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_label.cx.html',
        subject='feature.label',
        access=gws.PUBLIC,
        uid='gws.base.layer.templates.feature_label',
    ),
]

_DEFAULT_TILE_SIZE = 256


class Object(gws.Node, gws.ILayer):
    parent: gws.ILayer

    clientOptions: ClientOptions
    cssSelector: str

    canRenderBox = False
    canRenderXyz = False
    canRenderSvg = False

    supportsRasterServices = False
    supportsVectorServices = False

    isSearchable = False

    defaultBounds: gws.Bounds
    defaultResolutions: list[float]

    def configure(self):
        self.clientOptions = self.cfg('clientOptions')
        self.cssSelector = self.cfg('cssSelector')
        self.displayMode = self.cfg('display')
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.imageFormat = self.cfg('imageFormat')
        self.opacity = self.cfg('opacity')
        self.title = self.cfg('title')

        self.bounds = self.defaultBounds = self.cfg('_defaultBounds') or gws.gis.crs.WEBMERCATOR_BOUNDS
        self.resolutions = self.defaultResolutions = self.cfg('_defaultResolutions') or gws.gis.zoom.OSM_RESOLUTIONS

        self.templates = []
        self.models = []
        self.finders = []

        self.metadata = gws.Metadata()
        self.legend = None

        self.layers = []

        self.grid = None
        self.cache = None

        setattr(self, 'provider', None)
        self.sourceLayers = []

    def configure_layer(self):
        """Layer configuration protocol."""
        self.configure_provider()
        self.configure_sources()
        self.configure_models()
        self.configure_bounds()
        self.configure_resolutions()
        self.configure_grid()
        self.configure_legend()
        self.configure_cache()
        self.configure_metadata()
        self.configure_templates()
        self.configure_search()

    def post_configure(self):
        self.isSearchable = bool(self.finders)

    ##

    def configure_bounds(self):
        p = self.cfg('extent')
        if p:
            self.bounds = gws.Bounds(
                crs=self.defaultBounds.crs,
                extent=gws.gis.extent.from_list(p))
            return True

    def configure_cache(self):
        if not self.cfg('options.cache'):
            return True
        self.cache = gws.LayerCache(self.cfg('cache'))
        return True

    def configure_grid(self):
        p = self.cfg('grid')
        if p:
            if p.crs and p.crs != self.bounds.crs:
                raise gws.Error(f'invalid target grid crs')
            self.grid = gws.TileGrid(
                origin=p.origin or gws.Origin.nw,
                tileSize=p.tileSize or _DEFAULT_TILE_SIZE,
                bounds=gws.Bounds(crs=self.bounds.crs, extent=p.extent),
                resolutions=p.resolutions)
            return True

    def configure_legend(self):
        if not self.cfg('options.legend'):
            return True
        p = self.cfg('legend')
        if p:
            self.legend = self.create_child(gws.ext.object.legend, p)
            return True

    def configure_metadata(self):
        p = self.cfg('metadata')
        if p:
            self.metadata = gws.lib.metadata.from_config(p)
            return True

    def configure_models(self):
        p = self.cfg('models')
        if p:
            self.models = gws.compact(self.configure_model(c) for c in p)
            return True

    def configure_model(self, cfg):
        return self.create_child(gws.ext.object.model, cfg)

    def configure_provider(self):
        pass

    def configure_resolutions(self):
        p = self.cfg('zoom')
        if p:
            self.resolutions = gws.gis.zoom.resolutions_from_config(p, self.cfg('_defaultResolutions'))
            if not self.resolutions:
                raise gws.Error(f'layer {self.uid!r}: no resolutions, config={p!r} parent={self.defaultResolutions!r}')
            return True

    def configure_search(self):
        if not self.cfg('options.search'):
            return True
        p = self.cfg('finders')
        if p:
            self.finders = gws.compact(self.configure_finder(c) for c in p)
            return True

    def configure_finder(self, cfg):
        return self.create_child(gws.ext.object.finder, cfg)

    def configure_sources(self):
        pass

    def configure_templates(self):
        p = self.cfg('templates')
        if p:
            self.templates = gws.compact(self.configure_template(cfg) for cfg in p)
        for cfg in _DEFAULT_TEMPLATES:
            self.templates.append(self.root.create_shared(gws.ext.object.template, cfg))
        return True

    def configure_template(self, cfg):
        return self.create_child(gws.ext.object.template, cfg)

    def configure_group_layers(self, layer_configs):
        ls = []

        for cfg in layer_configs:
            cfg = gws.merge(
                cfg,
                _defaultBounds=self.bounds,
                _defaultResolutions=self.resolutions,
            )
            ls.append(self.create_child(gws.ext.object.layer, cfg))

        self.layers = gws.compact(ls)

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
        p = Props(
            clientOptions=self.clientOptions,
            cssSelector=self.cssSelector,
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
                origin=self.grid.origin,
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

    def render(self, lri):
        pass

    def get_features(self, search, user, views=None, model_uid=None):
        return []

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
