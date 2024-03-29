"""Base layer object."""
from typing import Optional

import gws
import gws.base.legend
import gws.base.model
import gws.base.search
import gws.base.template
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.gis.extent
import gws.gis.source
import gws.gis.zoom
import gws.lib.metadata
import gws.lib.style
import gws.lib.svg
import gws.lib.xmlx

import gws.types as t

DEFAULT_TILE_SIZE = 256


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


class AutoLayersOptions(gws.ConfigWithAccess):
    """Configuration for automatic layers."""

    applyTo: t.Optional[gws.gis.source.LayerFilter]
    config: 'Config'


class OwsConfig(gws.Config):
    layerName: str = ''
    featureName: str = ''
    geometryName: str = ''
    xmlNamespace: t.Optional[gws.XmlNamespace]


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: bool = False
    """the layer is expanded in the list view"""
    unlisted: bool = False
    """the layer is hidden in the list view"""
    selected: bool = False
    """the layer is initially selected"""
    hidden: bool = False
    """the layer is initially hidden"""
    unfolded: bool = False
    """the layer is not listed, but its children are"""
    exclusive: bool = False
    """only one of this layer's children is visible at a time"""


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
    """search providers"""
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
    ows: t.Optional[OwsConfig]
    """configuration for OWS services"""
    templates: t.Optional[list[gws.ext.config.template]]
    """layer templates"""
    title: str = ''
    """layer title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """layer resolutions and scales"""
    withSearch: t.Optional[bool] = True
    """layer is searchable"""
    withLegend: t.Optional[bool] = True
    """layer has a legend"""
    withCache: t.Optional[bool] = False
    """layer is cached"""
    withOws: t.Optional[bool] = True
    """layer is enabled for OWS services"""


class GridProps(gws.Props):
    origin: str
    extent: gws.Extent
    resolutions: list[float]
    tileSize: int


class Props(gws.Props):
    clientOptions: gws.LayerClientOptions
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


class Object(gws.Node, gws.ILayer):
    parent: gws.ILayer

    clientOptions: gws.LayerClientOptions
    cssSelector: str

    canRenderBox = False
    canRenderSvg = False
    canRenderXyz = False

    isEnabledForOws = False
    isGroup = False
    isSearchable = False

    hasLegend = False

    parentBounds: gws.Bounds
    parentResolutions: list[float]

    def configure(self):
        self.clientOptions = self.cfg('clientOptions') or gws.Data()
        self.cssSelector = self.cfg('cssSelector')
        self.displayMode = self.cfg('display')
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.imageFormat = self.cfg('imageFormat')
        self.opacity = self.cfg('opacity')
        self.title = self.cfg('title')

        self.parentBounds = self.cfg('_parentBounds')
        self.parentResolutions = self.cfg('_parentResolutions')
        self.mapCrs = self.parentBounds.crs

        self.bounds = self.parentBounds
        self.resolutions = self.parentResolutions

        self.templates = []
        self.models = []
        self.finders = []

        self.metadata = gws.Metadata()
        self.legend = None
        self.legendUrl = ''

        self.layers = []

        self.grid = None
        self.cache = None
        self.owsOptions = None

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
        self.configure_ows()

    ##

    def configure_bounds(self):
        p = self.cfg('extent')
        if p:
            self.bounds = gws.Bounds(
                crs=self.mapCrs,
                extent=gws.gis.extent.from_list(p))
            return True

    def configure_cache(self):
        if not self.cfg('withCache'):
            return True
        self.cache = gws.LayerCache(self.cfg('cache'))
        return True

    def configure_grid(self):
        p = self.cfg('grid')
        if p:
            if p.crs and p.crs != self.bounds.crs:
                raise gws.Error(f'layer {self!r}: invalid target grid crs')
            self.grid = gws.TileGrid(
                origin=p.origin or gws.Origin.nw,
                tileSize=p.tileSize or DEFAULT_TILE_SIZE,
                bounds=gws.Bounds(crs=self.bounds.crs, extent=p.extent),
                resolutions=p.resolutions)
            return True

    def configure_legend(self):
        if not self.cfg('withLegend'):
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
        return gws.config.util.configure_models(self)

    def configure_provider(self):
        pass

    def configure_resolutions(self):
        p = self.cfg('zoom')
        if p:
            self.resolutions = gws.gis.zoom.resolutions_from_config(p, self.cfg('_parentResolutions'))
            if not self.resolutions:
                raise gws.Error(f'layer {self!r}: no resolutions, config={p!r} parent={self.parentResolutions!r}')
            return True

    def configure_search(self):
        if not self.cfg('withSearch'):
            return True
        return gws.config.util.configure_finders(self)

    def configure_sources(self):
        pass

    def configure_templates(self):
        return gws.config.util.configure_templates(self)

    def configure_group_layers(self, layer_configs):
        ls = []

        for cfg in layer_configs:
            cfg = gws.merge(
                cfg,
                _parentBounds=self.bounds,
                _parentResolutions=self.resolutions,
            )
            ls.append(self.create_child(gws.ext.object.layer, cfg))

        self.layers = gws.compact(ls)

    def configure_ows(self):
        self.isEnabledForOws = self.cfg('withOws')

        p = self.cfg('ows') or gws.Data()
        self.owsOptions = gws.LayerOwsOptions(
            layerName=p.layerName or gws.to_uid(self.title),
            featureName=p.featureName or gws.to_uid(self.title),
            geometryName=p.geometryName or 'geometry',
            xmlNamespace=p.xmlNamespace,
        )

    ##

    def post_configure(self):
        self.isSearchable = bool(self.finders)
        self.hasLegend = bool(self.legend)

        if self.bounds.crs != self.mapCrs:
            raise gws.Error(f'layer {self!r}: invalid CRS {self.bounds.crs}')

        if not gws.gis.bounds.intersect(self.bounds, self.parentBounds):
            gws.log.warning(f'layer {self!r}: bounds outside of the parent bounds')
            self.bounds = gws.gis.bounds.copy(self.parentBounds)

        self.wgsExtent = gws.gis.bounds.transform(self.bounds, gws.gis.crs.WGS84).extent

        if self.legend:
            self.legendUrl = self.url_path('legend')

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

    def get_features_for_view(self, search, user, view_names= None):
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
