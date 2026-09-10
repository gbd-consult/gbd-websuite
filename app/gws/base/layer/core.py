"""Base layer object."""

from typing import Optional, cast

import gws
import gws.base.model
import gws.config.util
import gws.lib.crs
import gws.lib.grid
import gws.lib.image
import gws.lib.extent
import gws.gis.source
import gws.gis.zoom
import gws.base.metadata
import gws.base.grabber
import gws.lib.mime
import gws.lib.image
import gws.gis.cache

from . import ows

DEFAULT_TILE_SIZE = 256


class AutoLayersOptions(gws.ConfigWithAccess):
    """Configuration for automatic layers."""

    applyTo: Optional[gws.gis.source.LayerFilter]
    """Source layers to apply the configuration to."""
    config: dict
    """Configuration for the matching layers."""


class ClientOptions(gws.Data):
    """Client options for a layer."""

    expanded: bool = False
    """The layer is expanded in the list view."""
    unlisted: bool = False
    """The layer is hidden in the list view."""
    selected: bool = False
    """The layer is initially selected."""
    hidden: bool = False
    """The layer is initially hidden."""
    unfolded: bool = False
    """The layer is not listed, but its children are."""
    exclusive: bool = False
    """Only one of this layer children is visible at a time."""
    treeClassName = ''
    """CSS class name for the layer tree item."""


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    cache: Optional[gws.gis.cache.LayerConfig]
    """Cache configuration."""
    clientOptions: Optional[ClientOptions]
    """Options for the layer display in the client."""
    cssSelector: str = ''
    """Css selector for feature layers."""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.box
    """Layer display mode."""
    extent: Optional[gws.Extent]
    """Layer extent."""
    zoomExtent: Optional[gws.Extent]
    """Layer zoom extent."""
    extentBuffer: Optional[int]
    """Extent buffer."""
    finders: Optional[list[gws.ext.config.finder]]
    """Search providers."""
    grid: Optional[dict]
    """Client grid. (deprecated in 8.5)"""
    imageFormat: Optional[gws.lib.image.FormatConfig]
    """Image format."""
    legend: Optional[gws.ext.config.legend]
    """Legend configuration."""
    loadingStrategy: gws.FeatureLoadingStrategy = gws.FeatureLoadingStrategy.all
    """Feature loading strategy."""
    metadata: Optional[gws.base.metadata.Config]
    """Layer metadata."""
    models: Optional[list[gws.ext.config.model]]
    """Data models."""
    opacity: float = 1
    """Layer opacity."""
    ows: Optional[ows.Config]
    """Configuration for OWS services."""
    templates: Optional[list[gws.ext.config.template]]
    """Layer templates."""
    title: str = ''
    """Layer title."""
    zoom: Optional[gws.gis.zoom.Config]
    """Layer resolutions and scales."""
    withSearch: Optional[bool] = True
    """Layer is searchable."""
    withLegend: Optional[bool] = True
    """Layer has a legend."""
    withCache: Optional[bool] = True
    """Layer is cached. (changed in 8.5)"""
    withOws: Optional[bool] = True
    """Layer is enabled for OWS services."""


class Props(gws.Props):
    clientOptions: gws.LayerClientOptions
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


_DEFAULT_IMAGE_FORMAT = gws.lib.image.FormatConfig(name='png8', mimeTypes=['image/png'], options={'mode': 'P'})

CACHE_NAME_LENGTH = 12


class Object(gws.Layer):
    parent: gws.Layer

    clientOptions: gws.LayerClientOptions
    cssSelector: str

    canRenderBox = False
    canRenderSvg = False
    canRenderXyz = False

    isEnabledForOws = False
    isGroup = False
    isSearchable = False

    hasLegend = False

    parentWgsExtent: gws.Extent
    parentResolutions: list[float]

    def configure(self):
        self.clientOptions = self.cfg('clientOptions') or gws.Data()
        self.cssSelector = self.cfg('cssSelector')
        self.displayMode = self.cfg('display')
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.opacity = self.cfg('opacity')
        self.title = self.cfg('title')

        p = self.cfg('imageFormat') or _DEFAULT_IMAGE_FORMAT
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
        self.legendUrl = ''

        self.layers = []
        self.sourceLayers = []

        self.grabbers = {}

    def configure_layer(self):
        """Layer configuration protocol."""

        self.configure_provider()
        self.configure_sources()
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

        ext = gws.lib.extent.intersection(self.wgsExtent, self.parentWgsExtent)
        if ext:
            ext = self.mapCrs.clip_extent(ext)
        if not ext:
            gws.log.warning(f'layer {self!r}: extent outside of the parent extent wgs={self.wgsExtent} parent={self.parentWgsExtent}')
            ext = self.mapCrs.clip_extent(self.parentWgsExtent)
        self.bounds = gws.Bounds(crs=self.mapCrs, extent=gws.lib.extent.transform_from_wgs(ext, self.mapCrs))
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

    def configure_group_layers(self, layer_configs):
        ls = []

        for cfg in layer_configs:
            cfg = gws.u.merge(
                cfg,
                _parentWgsExtent=self.wgsExtent,
                _mapCrs=self.mapCrs,
                _parentResolutions=self.resolutions,
            )
            ls.append(self.create_child(gws.ext.object.layer, cfg))

        self.layers = gws.u.compact(ls)

    def configure_ows(self):
        self.isEnabledForOws = self.cfg('withOws', default=True)
        self.ows = self.create_child(ows.Object, self.cfg('ows'), _defaultName=gws.u.to_uid(self.title))

    ##

    def post_configure(self):
        self.isSearchable = bool(self.finders)
        self.hasLegend = bool(self.legend)

        self.zoomBounds = self.zoomBounds or self.bounds

        if self.legend:
            self.legendUrl = self.url_path('legend')

        self.post_configure_grabbers()

    def post_configure_grabbers(self):
        if not (self.canRenderBox or self.canRenderXyz):
            return

        p = cast(
            gws.gis.cache.LayerConfig,
            self.cfg('cache') or self.root.specs.read({}, 'gws.gis.cache.core.LayerConfig'),
        )
        cache_proto = gws.MapCache(
            name=p.name,
            maxAge=p.maxAge,
            maxLevel=p.maxLevel,
            requestBuffer=p.requestBuffer,
            requestTiles=p.requestTiles,
        )
        if not self.cfg('withCache'):
            cache_proto.maxAge = 0
        if not cache_proto.name:
            cache_proto.name = self.create_cache_name(cache_proto)

        cache_srids = []
        if p.crs:
            cache_srids = [gws.lib.crs.require(c).srid for c in p.crs]

        for crs in self.root.app.supported_crs():
            ext = crs.clip_extent(self.wgsExtent)
            if not ext:
                gws.log.warning(f'layer {self!r}: extent {self.wgsExtent} is incompatible with {crs!r}')
                continue
            cache = gws.MapCache(**vars(cache_proto))
            if cache_srids and crs.srid not in cache_srids:
                cache.maxAge = 0
            cache.name += f'_{crs.srid}'

            opts = gws.base.grabber.Options(
                crs=crs,
                cache=cache,
                extent=gws.lib.extent.transform_from_wgs(ext, crs),
                imageFormat=self.imageFormat,
                provider=getattr(self, 'serviceProvider', None),
            )

            gr = self.create_grabber(opts)
            if gr:
                self.grabbers[crs.srid] = gr

    def create_cache_name(self, cache: gws.MapCache) -> str:
        prov = getattr(self, 'serviceProvider', None)
        return gws.u.sha256(
            [
                prov.cache_hash() if prov else '',
                [sl.name for sl in self.sourceLayers],
                vars(self.imageFormat),
                list(self.wgsExtent),
                cache.requestBuffer,
                cache.requestTiles,
            ]
        )[:CACHE_NAME_LENGTH]

    def create_grabber(self, opts: gws.base.grabber.Options) -> Optional[gws.Grabber]:
        pass

    ##

    # @TODO use Node.find_ancestors

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

    def url_path(self, kind):
        ext = gws.lib.mime.extension_for(self.imageFormat.mimeTypes[0])
        url_path_suffix = '/gws.' + ext

        # layer urls, handled by the map action (base/map/action.py)
        if kind == 'box':
            return gws.u.action_url_path('mapGetBox', layerUid=self.uid) + url_path_suffix
        if kind == 'tile':
            return gws.u.action_url_path('mapGetXYZ', layerUid=self.uid) + '/z/{z}/x/{x}/y/{y}' + url_path_suffix
        if kind == 'legend':
            return gws.u.action_url_path('mapGetLegend', layerUid=self.uid) + url_path_suffix
        if kind == 'features':
            return gws.u.action_url_path('mapGetFeatures', layerUid=self.uid)

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
            p.url = self.url_path('tile')

        if self.displayMode == gws.LayerDisplayMode.box:
            p.type = 'box'
            p.url = self.url_path('box')

        return p

    def find_features(self, search, user):
        return []

    def render(self, lri):
        if lri.type == gws.LayerRenderInputType.box:
            return self.render_box(lri)
        if lri.type == gws.LayerRenderInputType.xyz:
            return self.render_tile(lri)
        if lri.type == gws.LayerRenderInputType.svg:
            return self.render_svg(lri)

    def grabber_for(self, lri: gws.LayerRenderInput) -> Optional[gws.Grabber]:
        crs = lri.targetCrs or self.mapCrs
        return self.grabbers.get(crs.srid)

    def render_tile(self, lri):
        gr = self.grabber_for(lri)
        if not gr:
            return
        return gws.LayerRenderOutput(
            content=gr.get_tile_as_bytes((lri.x, lri.y, lri.z), lri.renderParams),
        )

    def render_box(self, lri):
        gr = self.grabber_for(lri)
        if not gr:
            return

        params = lri.renderParams
        w, h = lri.view.pxSize

        if not lri.view.rotation:
            content = gr.get_box_as_bytes(lri.view.bounds.extent, w, h, params)
            return gws.LayerRenderOutput(content=content)

        circ = gws.lib.extent.circumsquare(lri.view.bounds.extent)
        d = gws.u.to_rounded_int(gws.lib.extent.diagonal((0, 0, w, h)))

        content = gr.get_box_as_bytes(circ, d, d, params)

        img = gws.lib.image.from_bytes(content)
        img.rotate(-lri.view.rotation).crop(
            (
                d / 2 - w / 2,
                d / 2 - h / 2,
                d / 2 + w / 2,
                d / 2 + h / 2,
            )
        )

        content = img.to_bytes(self.imageFormat.mimeTypes[0], self.imageFormat.options)
        return gws.LayerRenderOutput(content=content)

    def render_svg(self, lri):
        pass

    def render_legend(self, args=None) -> Optional[gws.LegendRenderOutput]:
        if not self.legend:
            return None

        def _get():
            out = self.legend.render()
            return out

        if not args:
            return gws.u.get_server_global('legend_' + self.uid, _get)

        return self.legend.render(args)
