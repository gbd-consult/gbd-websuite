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
import gws.gis.ows
import gws.gis.proj
import gws.gis.shape
import gws.gis.source
import gws.gis.zoom
import gws.tools.net
import gws.tools.units

import gws.types as t
from gws import cached_property

from . import types

_DEFAULT_STYLE_VALUES = {
    'fill': 'rgba(0,0,0,1)',
    'stroke': 'rgba(0,0,0,1)',
    'stoke_width': 1,
}

_DEFAULT_LEGEND_TEMPLATE = """
    <div class="legend"><img src="{path}"/></div>
"""

class Config(t.WithTypeAndAccess):
    """Layer configuration"""

    clientOptions: types.ClientOptions = {}  #: options for the layer display in the client
    dataModel: t.Optional[gws.common.model.Config]  #: layer data model
    description: t.Optional[t.ext.template.Config]  #: template for the layer description
    display: types.DisplayMode = 'box'  #: layer display mode
    edit: t.Optional[types.EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: feature formatting options
    legend: types.LegendConfig = {}  #: legend configuration
    meta: t.Optional[gws.common.metadata.Config]  #: layer meta data
    opacity: float = 1  #: layer opacity
    ows: t.Optional[types.OwsConfig]  #: OWS services options
    search: gws.common.search.Config = {}  #: layer search configuration
    title: str = ''  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class CustomConfig(t.WithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.gis.source.LayerFilter]  #: source layers this configuration applies to
    clientOptions: t.Optional[types.ClientOptions]  #: options for the layer display in the client
    dataModel: t.Optional[gws.common.model.Config]  #: layer data model
    description: t.Optional[t.ext.template.Config]  #: template for the layer description
    display: t.Optional[types.DisplayMode]  #: layer display mode
    edit: t.Optional[types.EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: feature formatting options
    legend: t.Optional[types.LegendConfig]  #: legend configuration
    meta: t.Optional[gws.common.metadata.Config]  #: layer meta data
    opacity: t.Optional[float]  #: layer opacity
    ows: t.Optional[types.OwsConfig]  #: OWS services options
    search: t.Optional[gws.common.search.Config]  #: layer search configuration
    title: t.Optional[str]  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


#:export ILayer
class Layer(gws.Object, t.ILayer):
    @property
    def props(self):
        return types.LayerProps(
            extent=self.extent if self.extent != self.map.extent else None,
            meta=self.meta,
            opacity=self.opacity,
            options=self.client_options,
            resolutions=self.resolutions if self.resolutions != self.map.resolutions else None,
            title=self.title,
            uid=self.uid,
        )

    @cached_property
    def description(self) -> str:
        ctx = {
            'layer': self,
        }
        return self.description_template.render(ctx).content

    @cached_property
    def has_search(self) -> bool:
        return len(self.get_children('gws.ext.search.provider')) > 0

    @cached_property
    def own_bounds(self) -> t.Optional[t.Bounds]:
        return

    @property
    def default_search_provider(self) -> t.Optional[t.ISearchProvider]:
        return

    def configure(self):
        super().configure()

        self.map: t.IMap = t.cast(t.IMap, self.get_closest('gws.common.map'))

        self.meta: t.MetaData = self.configure_metadata()
        self.title: str = self.meta.title

        uid = self.var('uid') or gws.as_uid(self.title) or 'layer'
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

        self.can_render_box: bool = False
        self.can_render_xyz: bool = False
        self.can_render_svg: bool = False

        self.is_public: bool = self.root.application.auth.get_role('all').can_use(self)
        self.is_editable: bool = False

        p = self.var('legend')
        self.has_legend: bool = p.enabled and p.url
        self.legend_url: str = p.url

        self.image_format: str = self.var('imageFormat')
        self.display: str = self.var('display')

        #:noexport
        self.cache: types.CacheConfig = self.var('cache')
        self.has_cache: bool = self.cache and self.cache.enabled

        #:noexport
        self.grid: types.GridConfig = self.var('grid')

        self.cache_uid: str = ''
        self.grid_uid: str = ''

        self.layers: t.List[t.ILayer] = []

        p = self.var('description')
        self.description_template: t.ITemplate = (
            self.root.create_object('gws.ext.template', p) if p
            else self.root.create_shared_object(
                'gws.ext.template',
                'default_layer_description',
                gws.common.template.builtin_config('layer_description')
            )
        )

        p = self.var('featureFormat')
        self.feature_format: t.IFormat = (
            self.root.create_object('gws.common.format', p) if p
            else self.root.create_shared_object(
                'gws.common.format',
                'default_feature_description',
                gws.common.template.builtin_config('feature_format')
            )
        )

        p = self.var('dataModel')
        self.data_model: t.Optional[t.IModel] = (self.create_child('gws.common.model', p) if p else None)

        self.resolutions: t.List[float] = gws.gis.zoom.resolutions_from_config(
            self.var('zoom'),
            self.map.resolutions)
        self.extent: t.Optional[t.Extent] = None

        self.opacity: float = self.var('opacity')
        self.client_options = self.var('clientOptions')

        self.geometry_type: t.Optional[t.GeometryType] = None

        p = self.var('style')
        self.style: t.IStyle = (
            gws.common.style.from_config(p) if p
            else gws.common.style.from_props(t.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES))
        )

        self.supports_wms: bool = False
        self.supports_wfs: bool = False

        self.ows_name: str = gws.as_uid(self.var('ows.name')) or self.uid.split('.')[-1]
        self.ows_services_enabled: t.List[str] = self.var('ows.servicesEnabled', default=[])
        self.ows_services_disabled: t.List[str] = self.var('ows.servicesDisabled', default=[])

        self.crs: str = self.var('crs') or self.map.crs

        p = self.var('editDataModel')
        self.edit_data_model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None
        self.edit_options: t.Data = self.var('edit')

        p = self.var('editStyle')
        self.edit_style: t.Optional[t.IStyle] = gws.common.style.from_config(p) if p else None

    def post_configure(self):
        super().post_configure()

        self.configure_search()
        self.configure_spatial_metadata()

    def configure_metadata(self, provider_meta=None) -> t.MetaData:
        """Load metadata from the config or from a provider, whichever comes first."""

        title = self.var('title')

        # use, in order 1) configured metadata, 2) provider meta, 3) dummy meta with title only
        m = self.var('meta') or provider_meta
        if not m:
            if title:
                m = t.MetaData(title=title)
            elif self.var('uid'):
                m = t.MetaData(title=self.var('uid'))
            else:
                m = t.MetaData()

        if title:
            # title at the top level config overrides meta title
            m.title = title

        meta = gws.common.metadata.from_config(m)
        p = t.cast(t.IProject, self.get_closest('gws.common.project'))
        if p:
            meta = gws.common.metadata.extend(meta, p.meta)
        return meta

    def configure_spatial_metadata(self):
        scales = [gws.tools.units.res2scale(r) for r in self.resolutions]
        self.meta.geographicExtent = gws.gis.extent.transform_to_4326(self.extent, self.map.crs)
        self.meta.minScale = int(min(scales))
        self.meta.maxScale = int(max(scales))
        self.meta.proj = gws.gis.proj.as_projection(self.map.crs)

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
                self.append_child(prov)
            return

        for cfg in p.providers:
            self.create_child('gws.ext.search.provider', cfg)

    def edit_access(self, user):
        # @TODO granular edit access

        if self.is_editable and self.edit_options and user.can_use(self.edit_options, parent=self):
            return ['all']

    def edit_operation(self, operation: str, feature_props: t.List[t.FeatureProps]) -> t.List[t.IFeature]:
        pass

    def props_for(self, user):
        p = super().props_for(user)
        if p:
            p['editAccess'] = self.edit_access(user)
        return p

    def mapproxy_config(self, mc):
        pass

    def render_box(self, rv: t.MapRenderView, client_params=None):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, rv: t.MapRenderView, style: t.IStyle = None):
        return None

    def render_legend(self) -> bytes:
        if self.legend_url.startswith('/'):
            return gws.read_file(self.legend_url, 'rb')
        return gws.gis.ows.request.raw_get(self.legend_url).content

    def render_html_legend(self) -> str:
        img = self.render_legend()
        path = gws.PRINT_DIR + f'/legend.{self.uid}.png'
        gws.write_file(path, img, 'wb')
        return _DEFAULT_LEGEND_TEMPLATE.replace('{path}', path)

    def get_features(self, bounds: t.Bounds, limit: int = 0) -> t.List[t.IFeature]:
        return []

    def ows_enabled(self, service: t.IOwsService) -> bool:
        if service.type == 'wms' and not self.supports_wms:
            return False
        if service.type == 'wfs' and not self.supports_wfs:
            return False
        if self.ows_services_disabled and service.uid in self.ows_services_disabled:
            return False
        if self.ows_services_enabled and service.uid not in self.ows_services_enabled:
            return False
        return True
