"""Base layer object."""

from typing import Optional, cast

import gws
import gws.base.model
import gws.config.util
import gws.lib.grid
import gws.lib.extent
import gws.gis.source
import gws.gis.zoom
import gws.base.metadata
import gws.lib.mime
import gws.lib.image
import gws.gis.cache

from . import ows

DEFAULT_IMAGE_FORMAT = gws.ImageFormat(name='png8', mimeTypes=['image/png'], options={'mode': 'P'})


class AutoLayersConfig(gws.ConfigWithAccess):
    """Configuration merged into automatically created layers that match a filter."""

    applyTo: Optional[gws.gis.source.LayerFilter]
    """Filter for the source layers the configuration applies to."""
    config: dict
    """Layer configuration to merge into each matching layer."""


class ClientConfig(gws.Data):
    """Display options for the layer in the client layer tree."""

    expanded: bool = False
    """Show the layer's children expanded in the layer tree."""
    unlisted: bool = False
    """Hide the layer from the layer tree."""
    selected: bool = False
    """Select the layer in the layer tree initially."""
    hidden: bool = False
    """Hide the layer on the map initially."""
    unfolded: bool = False
    """Show the layer's children in the tree instead of the layer itself."""
    exclusive: bool = False
    """Allow only one child layer to be visible at a time."""
    treeClassName = ''
    """CSS class for the layer's item in the layer tree."""


class ClientOptions(gws.Data):
    """Client options for a layer."""

    exclusive: bool
    """Only one of this layer children is visible at a time."""
    expanded: bool
    """A layer is expanded in the list view."""
    hidden: bool
    """A layer is initially hidden."""
    selected: bool
    """A layer is initially selected."""
    treeClassName: str
    """CSS class name for the layer tree item."""
    unfolded: bool
    """A layer is not listed, but its children are."""
    unlisted: bool
    """A layer is hidden in the list view."""


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    cache: Optional[gws.gis.cache.LayerConfig]
    """Tile cache options."""
    clientOptions: Optional[ClientConfig]
    """Display options in the client layer tree."""
    cssSelector: str = ''
    """CSS selector for styling the layer's features in the client."""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.box
    """The way the layer is rendered in the client."""
    extent: Optional[gws.Extent]
    """Layer extent in map CRS coordinates, clipped to the parent extent."""
    zoomExtent: Optional[gws.Extent]
    """Extent to zoom to when zooming to the layer."""
    extentBuffer: Optional[int]
    """Buffer around the layer extent."""
    finders: Optional[list[gws.ext.config.finder]]
    """Search providers for the layer's features."""
    grid: Optional[dict]
    """Ignored. (deprecated in 8.5)"""
    imageFormat: Optional[gws.lib.image.FormatConfig]
    """Format of rendered and cached images."""
    legend: Optional[gws.ext.config.legend]
    """Custom legend or legend options for the layer."""
    loadingStrategy: gws.FeatureLoadingStrategy = gws.FeatureLoadingStrategy.all
    """When the client loads the layer's features."""
    metadata: Optional[gws.base.metadata.Config]
    """Layer metadata."""
    models: Optional[list[gws.ext.config.model]]
    """Data models for the layer's features."""
    opacity: float = 1
    """Layer opacity, from 0 to 1."""
    ows: Optional[ows.Config]
    """Layer names, namespace and allowed services for OWS services."""
    templates: Optional[list[gws.ext.config.template]]
    """Templates for the layer's description and feature info."""
    title: str = ''
    """Layer title, shown in the layer tree."""
    zoom: Optional[gws.gis.zoom.Config]
    """Scales or zoom levels at which the layer is visible."""
    withSearch: Optional[bool] = True
    """Include the layer in searches."""
    withLegend: Optional[bool] = True
    """Show a legend for the layer."""
    withCache: Optional[bool] = True
    """Store rendered tiles in the persistent cache. (changed in 8.5)"""
    withOws: Optional[bool] = True
    """Publish the layer in OWS services."""


class Props(gws.Props):
    clientOptions: ClientOptions
    cssSelector: str
    displayMode: str
    extent: Optional[gws.Extent]
    zoomExtent: Optional[gws.Extent]
    geometryType: Optional[gws.GeometryType]
    grid: gws.lib.grid.Props
    layers: Optional[list['Props']]
    loadingStrategy: gws.FeatureLoadingStrategy
    metadata: gws.base.metadata.Props
    model: Optional[gws.base.model.Props]
    opacity: Optional[float]
    resolutions: Optional[list[float]]
    title: str = ''
    type: str
    uid: str
    url: str = ''


class Object(gws.Layer):
    parent: gws.Layer

    clientOptions: ClientOptions

    canRenderBox = False
    canRenderSvg = False
    canRenderTile = False
    canRenderInClient = False

    isEnabledForOws = False
    isGroup = False
    isSearchable = False

    hasLegend = False

    parentWgsExtent: gws.Extent
    parentResolutions: list[float]

    def configure(self):
        if self.cfg('grid') is not None:
            self.root.config_warning('"layer.grid" is deprecated and ignored')

        self.clientOptions = self.cfg('clientOptions') or gws.Data()
        self.cssSelector = self.cfg('cssSelector')

        self.displayMode = self.cfg('display')
        if self.displayMode == gws.LayerDisplayMode.client and not self.canRenderInClient:
            raise gws.ConfigurationError(f'display mode "client" is not supported for this layer')

        self.loadingStrategy = self.cfg('loadingStrategy')
        self.opacity = self.cfg('opacity')
        self.title = self.cfg('title')

        self.imageFormat = DEFAULT_IMAGE_FORMAT
        p = self.cfg('imageFormat')
        if p:
            self.imageFormat = gws.ImageFormat(name=p.name, mimeTypes=p.mimeTypes, options=p.options or {})

        self.parentWgsExtent = self.cfg('_parentWgsExtent')
        self.parentResolutions = self.cfg('_parentResolutions')
        self.mapCrs = self.cfg('_mapCrs')

        self.wgsExtent = self.parentWgsExtent
        self.bounds = cast(gws.Bounds, None)
        self.zoomBounds = cast(gws.Bounds, None)
        self.resolutions = self.parentResolutions

        self.templates = []
        self.models = []
        self.finders = []

        self.metadata = gws.base.metadata.new()
        self.legend = None

        self.layers = []
        self.sourceLayers = []

        self.grabbers = {}

    def configure_layer(self):
        """Layer configuration protocol."""

        self.configure_provider()
        self.configure_sources()
        self.configure_group()
        self.configure_models()
        self.configure_extent()
        self.configure_bounds()
        self.configure_zoom_bounds()
        self.configure_resolutions()
        self.configure_legend()
        self.configure_metadata()
        self.configure_templates()
        self.configure_search()
        self.configure_ows()

    ##

    def configure_group(self):
        pass

    def configure_extent(self):
        p = self.cfg('extent')
        if p:
            ext = gws.lib.extent.from_list(p)
            if not ext or not gws.lib.extent.is_valid(ext):
                raise gws.ConfigurationError(f'invalid extent {p!r}')
            self.wgsExtent = gws.lib.extent.transform_to_wgs(ext, self.mapCrs)
            return True

    def configure_bounds(self):
        """Bounds in the map CRS: the WGS extent clipped to the parent extent and to the CRS."""

        extent = gws.lib.extent.intersection(self.wgsExtent, self.parentWgsExtent)
        if extent:
            extent = self.mapCrs.clip_wgs_extent(extent)
        if extent is None:
            self.root.config_warning(f'layer extent outside of the parent extent wgs={self.wgsExtent}')
            extent = self.mapCrs.clip_wgs_extent(self.parentWgsExtent)
        if extent is None:
            raise gws.ConfigurationError(f'layer extent could not be determined wgs={self.wgsExtent}')
        self.bounds = gws.Bounds(
            crs=self.mapCrs,
            extent=gws.lib.extent.transform_from_wgs(extent, self.mapCrs),
        )
        return True

    def configure_zoom_bounds(self):
        p = self.cfg('zoomExtent')
        if p:
            ext = gws.lib.extent.from_list(p)
            if not ext or not gws.lib.extent.is_valid(ext):
                raise gws.ConfigurationError(f'invalid extent {p!r}')
            self.zoomBounds = gws.Bounds(crs=self.mapCrs, extent=ext)
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
            self.metadata = gws.base.metadata.from_config(p)
            return True

    def configure_models(self):
        return gws.config.util.configure_models_for(self)

    def configure_provider(self):
        pass

    def configure_resolutions(self):
        p = self.cfg('zoom')
        if p:
            gws.gis.zoom.warn_deprecated_options(p, self.root)
            self.resolutions = gws.gis.zoom.resolutions_for_layer(p, self.cfg('_parentResolutions'), self.mapCrs)
            if not self.resolutions:
                raise gws.Error(f'layer {self!r}: no resolutions, config={p!r} parent={self.parentResolutions!r}')
            return True

    def configure_search(self):
        if not self.cfg('withSearch'):
            return True
        return gws.config.util.configure_finders_for(self)

    def configure_sources(self):
        pass

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self)

    def configure_ows(self):
        self.isEnabledForOws = self.cfg('withOws', default=True)
        self.ows = self.create_child(ows.Object, self.cfg('ows'), _defaultName=gws.u.to_uid(self.title))

    ##

    def post_configure(self):
        self.isSearchable = bool(self.finders)
        self.hasLegend = bool(self.legend)

        self.zoomBounds = self.zoomBounds or self.bounds

    ##

    def url_path_for(self, kind):
        ext = gws.lib.mime.extension_for(self.imageFormat.mimeTypes[0])
        url_path_suffix = '/gws.' + ext

        # layer urls, handled by the map action (base/map/action.py)
        if kind == 'box':
            return gws.u.action_url_path('mapGetBox', layerUid=self.uid) + url_path_suffix
        if kind == 'tile':
            return gws.u.action_url_path('mapGetTile', layerUid=self.uid) + '/z/{z}/x/{x}/y/{y}' + url_path_suffix
        if kind == 'legend':
            return gws.u.action_url_path('mapGetLegend', layerUid=self.uid) + url_path_suffix
        if kind == 'features':
            return gws.u.action_url_path('mapGetFeatures', layerUid=self.uid)
        raise gws.Error(f'invalid argument {kind=}')

    def props(self, user):
        p = Props(
            clientOptions=self.clientOptions,
            cssSelector=self.cssSelector,
            displayMode=self.displayMode,
            extent=self.bounds.extent,
            zoomExtent=self.zoomBounds.extent,
            layers=self.layers,
            loadingStrategy=self.loadingStrategy,
            metadata=gws.base.metadata.props(self.metadata),
            opacity=self.opacity,
            resolutions=sorted(self.resolutions, reverse=True),
            title=self.title,
            uid=self.uid,
        )

        gr = self.grabbers.get(self.mapCrs.srid)
        if gr:
            p.grid = gws.lib.grid.props_for_resolutions(gr.grid, self.resolutions)

        if self.displayMode == gws.LayerDisplayMode.tile:
            p.type = 'tile'
            p.url = self.url_path_for('tile')

        if self.displayMode == gws.LayerDisplayMode.box:
            p.type = 'box'
            p.url = self.url_path_for('box')

        return p

    def find_features(self, search, user):
        return []

    def render(self, lri):
        if lri.type == gws.LayerRenderInputType.box:
            return self.render_box(lri)
        if lri.type == gws.LayerRenderInputType.tile:
            return self.render_tile(lri)
        if lri.type == gws.LayerRenderInputType.svg:
            return self.render_svg(lri)

    def render_box(self, lri):
        pass

    def render_tile(self, lri):
        pass

    def render_svg(self, lri):
        pass

    def render_legend(self, args=None):
        if not self.legend:
            return

        legend = self.legend
        if not args:
            return gws.u.get_server_global('legend_' + self.uid, legend.render)
        return legend.render(args)
