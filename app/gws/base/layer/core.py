import re

import gws
import gws.base.map.action
import gws.base.metadata
import gws.base.model
import gws.base.search
import gws.base.template
import gws.lib.legend
import gws.lib.extent
import gws.lib.gis
import gws.lib.img
import gws.lib.ows
import gws.lib.style
import gws.lib.svg
import gws.lib.units
import gws.lib.zoom
import gws.types as t

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
    reqSize: int = 0  #: number of metatiles to fetch
    reqBuffer: int = 0  #: pixel buffer


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

    clientOptions: ClientOptions = {}  #: options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: DisplayMode = DisplayMode.box  #: layer display mode
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: LegendConfig = {}  #: legend configuration
    metaData: t.Optional[gws.base.metadata.Config]  #: layer metadata
    opacity: float = 1  #: layer opacity
    ows: OwsConfig = {}  #: OWS services options
    search: gws.base.search.Config = {}  #: layer search configuration
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: client templates
    title: str = ''  #: layer title
    zoom: t.Optional[gws.lib.zoom.Config]  #: layer resolutions and scales


class CustomConfig(gws.WithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.lib.gis.LayerFilter]  #: source layers this configuration applies to
    clientOptions: t.Optional[ClientOptions]  # options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: t.Optional[DisplayMode]  #: layer display mode
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: t.Optional[LegendConfig]  # #: legend configuration
    metaData: t.Optional[gws.base.metadata.Config]  #: layer metadata
    opacity: t.Optional[float]  #: layer opacity
    ows: t.Optional[OwsConfig]  #: OWS services options
    search: t.Optional[gws.base.search.Config]  #: layer search configuration
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: client templates
    title: t.Optional[str]  #: layer title
    zoom: t.Optional[gws.lib.zoom.Config]  #: layer resolutions and scales


class Props(gws.Props):
    dataModel: t.Optional[gws.base.model.Props]
    editAccess: t.Optional[t.List[str]]
    editStyle: t.Optional[gws.lib.style.Props]
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    metaData: gws.base.metadata.Props
    opacity: t.Optional[float]
    options: ClientOptions
    resolutions: t.Optional[t.List[float]]
    style: t.Optional[gws.lib.style.Props]
    tileSize: int = 0
    title: str = ''
    type: str
    uid: str
    url: str = ''


class ConfiguredFlags(gws.Data):
    extent: bool
    layers: bool
    legend: bool
    metadata: bool
    resolutions: bool
    search: bool


class Object(gws.Node, gws.ILayer):
    map: gws.IMap

    can_render_box: bool = False
    can_render_xyz: bool = False
    can_render_svg: bool = False

    is_group: bool = False
    is_public: bool = False
    is_editable: bool = False

    supports_wms: bool = False
    supports_wfs: bool = False

    cache: CacheConfig
    cache_uid: str
    client_options: gws.Data
    crs: gws.Crs
    data_model: t.Optional[gws.IDataModel]
    description_template: gws.ITemplate
    display: str
    edit_data_model: t.Optional[gws.IDataModel]
    edit_options: t.Optional[gws.Data]
    edit_style: t.Optional[gws.IStyle]
    extent: gws.Extent
    geometry_type: t.Optional[gws.GeometryType]
    grid: GridConfig
    grid_uid: str
    image_format: str
    layers: t.List[gws.ILayer]
    legend: gws.Legend
    legend_output: t.Optional[gws.LegendRenderOutput]
    metadata: gws.IMetaData
    opacity: float
    ows_enabled: bool
    ows_enabled_services_pattern: gws.Regex
    ows_enabled_services_uids: t.List[str]
    ows_feature_name: str
    ows_name: str
    resolutions: t.List[float]
    search_providers: t.List[gws.ISearchProvider]
    style: t.Optional[gws.IStyle]
    templates: gws.ITemplateBundle

    has_configured: ConfiguredFlags

    @property
    def props(self):
        return gws.Props(
            extent=self.extent,
            metaData=self.metadata,
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
        return gws.base.map.action.url_for_get_legend(self.uid)

    @property
    def ancestors(self) -> t.List[gws.ILayer]:
        ps = []
        p = self.parent
        while p.is_a('gws.ext.layer'):
            ps.append(t.cast(gws.ILayer, p))
            p = p.parent
        return ps

    def configure(self):
        self.has_configured = ConfiguredFlags()

        self.map = t.cast(gws.IMap, self.get_closest('gws.base.map'))

        uid = self.var('uid') or gws.as_uid(self.var('title'))
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

        self.is_public = self.root.application.auth.get_role('all').can_use(self)
        self.cache = self.var('cache', default=CacheConfig(enabled=False))
        self.cache_uid = ''
        self.client_options = self.var('clientOptions')
        self.crs = self.var('crs') or (self.map.crs if self.map else gws.EPSG_3857)
        self.display = self.var('display')
        self.edit_options = self.var('edit')
        self.geometry_type = None
        self.grid = self.var('grid', default=GridConfig())
        self.grid_uid = ''
        self.image_format = self.var('imageFormat')
        self.layers = []
        self.legend = gws.Legend(enabled=False)
        self.legend_output = None
        self.metadata = t.cast(gws.IMetaData, None)
        self.opacity = self.var('opacity')
        self.ows_enabled = self.var('ows.enabled')
        self.ows_enabled_services_uids = self.var('ows.enabledServices.uids') or []
        self.ows_enabled_services_pattern = self.var('ows.enabledServices.pattern')
        self.ows_feature_name = ''
        self.ows_name = ''
        self.resolutions = []
        self.search_providers = []

        p = self.var('dataModel')
        self.data_model = self.create_child(gws.base.model.Object, p) if p else None

        p = self.var('editDataModel')
        self.edit_data_model = self.create_child(gws.base.model.Object, p) if p else None

        p = self.var('templates')
        self.templates = t.cast(
            gws.base.template.Bundle,
            self.create_child(gws.base.template.Bundle, gws.Config(templates=p, defaults=gws.base.template.BUILTINS)))
        self.description_template = self.templates.find(subject='layer.description')

        p = self.var('style') or _DEFAULT_STYLE
        self.style = t.cast(gws.IStyle, self.create_child(gws.lib.style.Object, p))

        p = self.var('editStyle')
        self.edit_style = self.create_child(gws.lib.style.Object, p) if p else self.style

        p = self.var('metaData')
        if p:
            self.configure_metadata_from(p)
            self.has_configured.metadata = True

        p = self.var('extent')
        if p:
            self.extent = gws.lib.extent.from_list(p)
            if not self.extent:
                raise gws.Error(f'invalid extent {p!r} in layer={self.uid!r}')
            self.has_configured.extent = True

        p = self.var('zoom')
        if p:
            self.resolutions = gws.lib.zoom.resolutions_from_config(p, self.map.resolutions if self.map else [])
            if not self.resolutions:
                raise gws.Error(f'invalid zoom configuration in layer={self.uid!r}')
            self.has_configured.resolutions = True

        p = self.var('search')
        if p:
            if not p.enabled:
                self.search_providers = []
                self.has_configured.search = True
            elif p.providers:
                self.search_providers = [
                    t.cast(gws.ISearchProvider, self.create_child('gws.ext.search.provider', c))
                    for c in p.providers]
                self.has_configured.search = True

        p = self.var('legend')
        if p:
            if not p.enabled:
                self.legend = gws.Legend(enabled=False)
                self.has_configured.legend = True
            elif p.path:
                self.legend = gws.Legend(enabled=True, path=p.path, options=p.options or {})
                self.has_configured.legend = True
            elif p.url:
                self.legend = gws.Legend(enabled=True, url=p.url, options=p.options or {})
                self.has_configured.legend = True
            elif p.template:
                tpl = self.create_child('gws.ext.template', p.template)
                self.legend = gws.Legend(enabled=True, template=tpl, options=p.options or {})
                self.has_configured.legend = True

    def configure_metadata_from(self, m: gws.base.metadata.Values):
        self.metadata = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, m))
        self.title = self.var('title') or self.metadata.title
        self.ows_name = self.var('ows.name') or self.uid.split('.')[-1]
        self.ows_feature_name = self.var('ows.featureName') or self.ows_name
        self.has_configured.metadata = True

    def post_configure(self):
        if not self.resolutions:
            if self.map:
                self.resolutions = self.map.resolutions

        if not self.resolutions:
            raise gws.Error(f'no resolutions defined in layer={self.uid!r}')

        if not self.metadata:
            self.configure_metadata_from(gws.base.metadata.Values(
                title=self.var('title') or self.var('uid') or 'layer'))

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

    _LEGEND_CACHE_LIFETIME = 3600

    def get_legend(self, context=None) -> t.Optional[gws.LegendRenderOutput]:
        """Render a legend and return the path to the legend image."""

        if not self.legend.enabled:
            return None

        def _get():
            out = self.render_legend(context)
            if not context and not out:
                self.legend.enabled = False
            return out

        if not context:
            return gws.get_cached_object('LEGEND_' + self.uid, _get, self._LEGEND_CACHE_LIFETIME)

        return self.render_legend(context)

    def render_legend(self, context=None):
        return gws.lib.legend.render(self.legend, context)

    def get_features(self, bounds: gws.Bounds, limit: int = 0) -> t.List[gws.IFeature]:
        return []

    def enabled_for_ows(self, service: gws.IOwsService) -> bool:
        if not self.ows_enabled:
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
