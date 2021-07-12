import re

import gws
import gws.types as t
import gws.base.map.action
import gws.lib.metadata
import gws.base.model
import gws.base.search
import gws.base.template
import gws.lib.ows
import gws.lib.source
import gws.lib.zoom
import gws.lib.style
import gws.lib.svg
import gws.lib.units

_DEFAULT_STYLE = gws.Config(
    type='css',
    values={
        'fill': 'rgba(0,0,0,1)',
        'stroke': 'rgba(0,0,0,1)',
        'stoke_width': 1,
    }
)

_DEFAULT_LEGEND_HTML = """<div class="legend"><img src="{path}"/></div>"""


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False  #: the layer is expanded in the list view
    listed: t.Optional[bool] = True  #: the layer is displayed in this list view
    selected: t.Optional[bool] = False  #: the layer is intially selected
    visible: t.Optional[bool] = True  #: the layer is intially visible
    unfolded: t.Optional[bool] = False  #: the layer is not listed, but its children are
    exclusive: t.Optional[bool] = False  #: only one of this layer's children is visible at a time


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'  #: png 8-bit
    png24 = 'png24'  #: png 24-bit


class DisplayMode(t.Enum):
    """Layer display mode"""

    box = 'box'  #: display a layer as one big image (WMS-alike)
    tile = 'tile'  #: display a layer in a tile grid
    client = 'client'  #: draw a layer in the client


class CacheConfig(gws.Config):
    """Cache configuration"""

    enabled: bool = False  #: cache is enabled
    maxAge: gws.Duration = '7d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache


class GridConfig(gws.Config):
    """Grid configuration for caches and tiled data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    metaSize: int = 0  #: number of meta-tiles to fetch
    metaBuffer: int = 0  #: pixel buffer


class EditConfig(gws.WithAccess):
    """Edit access for a layer"""

    pass


class LegendConfig(gws.Config):
    """Layer legend confuguration."""

    enabled: bool = True  #: the legend is enabled
    url: t.Optional[gws.Url]  #: url of the legend image
    path: t.Optional[gws.FilePath]  #: path of the legend image
    template: t.Optional[gws.ext.template.Config]  #: template for an HTML legend
    options: t.Optional[dict]  #: provider-dependent legend options


class FlattenConfig(gws.Config):
    """Layer hierarchy flattening"""

    level: int  #: flatten level
    useGroups: bool = False  #: use group names (true) or image layer names (false)


class OwsEnabledServicesConfig(gws.Config):
    """Configuration for enabled OWS services"""

    uids: t.Optional[t.List[str]]  #: enabled services uids
    pattern: gws.Regex = ''  #: pattern for enabled service uids


class OwsConfig(gws.Config):
    """OWS services confuguration"""

    name: t.Optional[str]  #: layer name for ows services
    featureName: t.Optional[str]  #: feature name for ows services
    enabled: bool = True  #: enable this layer for ows services
    enabledServices: t.Optional[OwsEnabledServicesConfig]  #: enabled OWS services


class Config(gws.WithAccess):
    """Layer configuration"""

    clientOptions: ClientOptions = {}  # type: ignore  #: options for the layer display in the client
    legend: LegendConfig = {}  # type: ignore  #: legend configuration
    ows: OwsConfig = {}  # type: ignore #: OWS services options
    search: gws.base.search.Config = {}  # type: ignore  #: layer search configuration

    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: DisplayMode = DisplayMode.box  #: layer display mode
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    meta: t.Optional[gws.lib.metadata.Config]  #: layer meta data
    opacity: float = 1  #: layer opacity
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: client templates
    title: str = ''  #: layer title
    zoom: t.Optional[gws.lib.zoom.Config]  #: layer resolutions and scales


# class CustomConfig(gws.WithAccess):
#     """Custom layer configuration"""
#
#     applyTo: t.Optional[gws.lib.source.Filter]  #: source layers this configuration applies to
#     clientOptions: t.Optional[ClientOptions]  #: options for the layer display in the client
#     dataModel: t.Optional[gws.base.model.Config]  #: layer data model
#     display: t.Optional[DisplayMode]  #: layer display mode
#     edit: t.Optional[EditConfig]  #: editing permissions
#     extent: t.Optional[gws.Extent]  #: layer extent
#     extentBuffer: t.Optional[int]  #: extent buffer
#     legend: t.Optional[LegendConfig]  #: legend configuration
#     meta: t.Optional[gws.lib.metadata.Config]  #: layer meta data
#     opacity: t.Optional[float]  #: layer opacity
#     ows: t.Optional[OwsConfig]  #: OWS services options
#     search: t.Optional[gws.base.search.Config]  #: layer search configuration
#     templates: t.Optional[t.List[gws.ext.template.Config]]  #:client templates
#     title: t.Optional[str]  #: layer title
#     zoom: t.Optional[gws.lib.zoom.Config]  #: layer resolutions and scales


class Props(gws.Props):
    editAccess: t.Optional[t.List[str]]
    editStyle: t.Optional[gws.lib.style.Props]
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    meta: gws.lib.metadata.Props
    opacity: t.Optional[float]
    options: ClientOptions
    resolutions: t.Optional[t.List[float]]
    style: t.Optional[gws.lib.style.Props]
    tileSize: int = 0
    title: str = ''
    type: str
    uid: str
    url: str = ''


class Object(gws.Node, gws.ILayer):
    map: gws.IMap
    meta: gws.IMeta

    can_render_box: bool = False
    can_render_xyz: bool = False
    can_render_svg: bool = False

    is_group: bool = False
    is_public: bool = False
    is_editable: bool = False

    supports_wms: bool = False
    supports_wfs: bool = False

    legend: gws.Legend

    image_format: str
    display: str

    layers: t.List[gws.ILayer] = []

    templates: gws.ITemplateBundle
    data_model: t.Optional[gws.IDataModel]
    style: t.Optional[gws.IStyle]
    search_providers: t.List[gws.ISearchProvider]

    resolutions: t.List[float]
    extent: gws.Extent
    opacity: float
    geometry_type: t.Optional[gws.GeometryType]
    crs: gws.Crs

    client_options: gws.Data

    ows_name: str
    ows_feature_name: str

    edit_data_model: t.Optional[gws.IDataModel]
    edit_options: t.Optional[gws.Data]
    edit_style: t.Optional[gws.IStyle]

    description_template: gws.ITemplate
    ows_enabled: bool
    ows_enabled_services_uids: t.List[str]
    ows_enabled_services_pattern: gws.Regex

    cache: CacheConfig
    grid: GridConfig
    cache_uid: str = ''
    grid_uid: str = ''

    @property
    def props(self):
        return gws.Props(
            extent=self.extent if self.extent != self.map.extent else None,
            meta=self.meta,
            opacity=self.opacity,
            options=self.client_options,
            resolutions=self.resolutions,
            title=self.title,
            uid=self.uid,
        )

    @property
    def description(self) -> str:
        context = {'layer': self}
        return self.description_template.render(context).content

    @property
    def has_cache(self) -> bool:
        return self.cache.enabled

    @property
    def has_search(self) -> bool:
        return len(self.search_providers) > 0

    @property
    def has_legend(self) -> bool:
        return self.legend.enabled

    @property
    def own_bounds(self) -> t.Optional[gws.Bounds]:
        return None

    @property
    def default_search_provider(self) -> t.Optional[gws.ISearchProvider]:
        return None

    @property
    def legend_url(self):
        return gws.base.map.action.url_for_render_legend(self.uid)

    @property
    def ancestors(self) -> t.List[gws.ILayer]:
        ps = []
        p = self.parent
        while p.is_a('gws.ext.layer'):
            ps.append(t.cast(gws.ILayer, p))
            p = p.parent
        return ps

    def configure(self):
        self.map = t.cast(gws.IMap, self.get_closest('gws.base.map'))
        self.crs = self.var('crs') or self.map.crs

        self.meta = self.configure_metadata()
        self.title = ''  ### self.meta.data.title

        uid = self.var('uid') or gws.as_uid(self.title) or 'layer'
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

        self.is_public = self.root.application.auth.get_role('all').can_use(self)

        self.legend = gws.Legend(enabled=False)

        self.image_format = self.var('imageFormat')
        self.display = self.var('display')

        self.cache = self.var('cache', default=CacheConfig(enabled=False))
        self.grid = self.var('grid', default=GridConfig())

        self.layers = []

        p = self.var('templates')
        self.templates = t.cast(
            gws.base.template.Bundle,
            self.create_child(gws.base.template.Bundle, gws.Config(templates=p, defaults=gws.base.template.BUILTINS)))
        self.description_template = t.cast(
            gws.base.template.Object,
            self.templates.find(subject='layer.description'))

        p = self.var('dataModel')
        self.data_model = self.create_child(gws.base.model.Object, p) if p else None

        self.resolutions = gws.lib.zoom.resolutions_from_config(self.var('zoom'), self.map.resolutions)
        if not self.resolutions:
            raise gws.Error(f'no resolutions')

        # NB: the extent will be configured later on in map._configure_extent
        self.extent = t.cast(gws.Extent, None)

        self.opacity = self.var('opacity')
        self.client_options = self.var('clientOptions')

        self.geometry_type = None

        p = self.var('style') or _DEFAULT_STYLE
        self.style = t.cast(gws.IStyle, self.create_child(gws.lib.style.Object, p))

        p = self.var('editDataModel')
        self.edit_data_model = self.create_child(gws.base.model.Object, p) if p else None
        self.edit_options = self.var('edit')

        p = self.var('editStyle')
        self.edit_style = self.create_child(gws.lib.style.Object, p) if p else self.style

        self.ows_name = self.var('ows.name') or self.uid.split('.')[-1]
        self.ows_feature_name = self.var('ows.featureName') or self.ows_name
        self.ows_enabled = self.var('ows.enabled')
        self.ows_enabled_services_uids = self.var('ows.enabledServices.uids') or []
        self.ows_enabled_services_pattern = self.var('ows.enabledServices.pattern')

    def post_configure(self):
        try:
            self.configure_search()
        except Exception as e:
            gws.log.exception()

        legend = self.configure_legend()
        if legend:
            self.legend = legend
            self.legend.options = self.var('legend.options', default={})

    def configure_metadata(self, provider_meta=None) -> gws.IMeta:
        """Load metadata from the config or from a provider, whichever comes first."""

        title = self.var('title')

        # use, in order 1) configured metadata, 2) provider meta, 3) dummy meta with title only
        m = self.var('meta')

        if not m and provider_meta:
            m = gws.Config(provider_meta)

        if not m:
            if title:
                m = gws.Config(title=title)
            elif self.var('uid'):
                m = gws.Config(title=self.var('uid'))
            else:
                m = gws.Config()

        if title:
            # title at the top level config overrides meta title
            m.title = title

        return t.cast(gws.IMeta, self.create_child(gws.lib.metadata.Object, m))

    def configure_search(self):
        # search can be
        # 1) missing = use default provider
        # 2) disabled (enabled=False) = skip
        # 3) just enabled = use default provider
        # 4) enabled with explicit providers = use these

        p = self.var('search')

        if p and not p.enabled:
            return

        if not p or not p.providers:
            prov = self.default_search_provider
            if prov:
                self.search_providers.append(prov)
            return

        for cfg in p.providers:
            self.search_providers.append(
                t.cast(gws.ISearchProvider, self.create_child('gws.ext.search.provider', cfg)))

    def edit_access(self, user):
        # @TODO granular edit access

        if self.is_editable and self.edit_options and user.can_use(self.edit_options, parent=self):
            return ['all']

    # def edit_operation(self, operation: str, feature_props: t.List[gws.lib.feature.Props]) -> t.List[gws.IFeature]:
    #     pass
    # 
    def props_for(self, user):
        p = super().props_for(user)
        if p:
            p.editAccess = self.edit_access(user)
        return p

    def mapproxy_config(self, mc):
        pass

    def render_box(self, rv: gws.MapRenderView, extra_params=None):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> str:
        return gws.lib.svg.as_xml(self.render_svg_tags(rv, style))

    def render_svg_tags(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> t.List[gws.Tag]:
        return []

    def configure_legend(self) -> gws.Legend:
        p = self.var('legend')
        if not p.enabled:
            return gws.Legend(enabled=False)

        if p.path:
            return gws.Legend(enabled=True, path=p.path)

        if p.url:
            return gws.Legend(enabled=True, url=p.url)

        if p.template:
            return gws.Legend(enabled=True, template=self.create_child('gws.ext.template', p.template))

    def render_legend_to_path(self, context=None) -> t.Optional[str]:
        """Render a legend and return the path to the legend image."""

        if self.legend.path:
            return self.legend.path

        cache_path = gws.LEGEND_CACHE_DIR + '/' + self.uid + '.png'

        if self.legend.url:
            try:
                r = gws.lib.ows.request.raw_get(self.legend.url)
            except gws.lib.ows.error.Error as e:
                gws.log.error(f'layer {self.uid!r}: legend download failed: {e!r}')
                self.legend.enabled = False
                return None

            gws.write_file_b(cache_path, r.content)
            self.legend.path = cache_path
            return self.legend.path

        if self.legend.template:
            self.legend.template.render(context, gws.TemplateRenderArgs(out_path=cache_path, format='png'))
            self.legend.path = cache_path
            return self.legend.path

        img = self.render_legend_to_image(context)
        if img:
            gws.write_file_b(cache_path, img)
            self.legend.path = cache_path
            return self.legend.path

        self.legend.enabled = False
        return None

    def render_legend_to_image(self, context=None) -> t.Optional[bytes]:
        pass

    def render_legend_to_html(self, context=None) -> t.Optional[str]:
        """Render a legend in the html format."""

        if self.legend.template:
            return self.legend.template.render(context).content

        path = self.render_legend_to_path(context)
        if path:
            return _DEFAULT_LEGEND_HTML.replace('{path}', path)

        return ''

    def get_features(self, bounds: gws.Bounds, limit: int = 0) -> t.List[gws.IFeature]:
        return []

    def enabled_for_ows(self, service: gws.IOwsService) -> bool:
        if not self.enabled_for_ows:
            return False
        if self.ows_enabled_services_uids:
            return service.uid in self.ows_enabled_services_uids
        if self.ows_enabled_services_pattern:
            return re.search(self.ows_enabled_services_pattern, service.uid) is not None
        if service.service_type == 'wms' and self.supports_wms:
            return True
        if service.service_type == 'wfs' and self.supports_wfs:
            return True
        if self.layers:
            return any(la.enabled_for_ows(service) for la in self.layers)
        return False


def add_layers_to_object(target: gws.Node, layer_configs):
    ls = []
    skip_invalid = target.var('skipInvalidLayers', with_parent=True)
    for p in layer_configs:
        try:
            ls.append(target.create_child('gws.ext.layer', p))
        except Exception as e:
            uid = gws.get(p, 'uid')
            gws.log.error(f'FAILED LAYER: parent={target.uid!r} layer={uid!r} error={e!r}')
            if skip_invalid:
                gws.log.exception()
            else:
                raise
    return ls
