import re

import gws

import gws.common.auth
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
import gws.tools.svg

import gws.types as t
from gws import cached_property

from . import types

_DEFAULT_STYLE_VALUES = {
    'fill': 'rgba(0,0,0,1)',
    'stroke': 'rgba(0,0,0,1)',
    'stoke_width': 1,
}

_DEFAULT_LEGEND_HTML = """<div class="legend"><img src="{path}"/></div>"""


class Config(t.WithTypeAndAccess):
    """Layer configuration"""

    clientOptions: types.ClientOptions = {}  #: options for the layer display in the client
    dataModel: t.Optional[gws.common.model.Config]  #: layer data model
    display: types.DisplayMode = 'box'  #: layer display mode
    edit: t.Optional[types.EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: types.LegendConfig = {}  #: legend configuration
    meta: t.Optional[gws.common.metadata.Config]  #: layer meta data
    opacity: float = 1  #: layer opacity
    ows: types.OwsConfig = {}  #: OWS services options
    search: gws.common.search.Config = {}  #: layer search configuration
    templates: t.Optional[t.List[t.ext.template.Config]]  #: client templates
    title: str = ''  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales
    order: t.Optional[int]  #: layer order


class CustomConfig(t.WithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.gis.source.LayerFilter]  #: source layers this configuration applies to
    clientOptions: t.Optional[types.ClientOptions]  #: options for the layer display in the client
    dataModel: t.Optional[gws.common.model.Config]  #: layer data model
    display: t.Optional[types.DisplayMode]  #: layer display mode
    edit: t.Optional[types.EditConfig]  #: editing permissions
    extent: t.Optional[t.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: t.Optional[types.LegendConfig]  #: legend configuration
    meta: t.Optional[gws.common.metadata.Config]  #: layer meta data
    opacity: t.Optional[float]  #: layer opacity
    ows: t.Optional[types.OwsConfig]  #: OWS services options
    search: t.Optional[gws.common.search.Config]  #: layer search configuration
    templates: t.Optional[t.List[t.ext.template.Config]]  #:client templates
    title: t.Optional[str]  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales
    order: t.Optional[int]  #: layer order
    cache: t.Optional[types.CacheConfig]  #: cache configuration


#:export
class LayerLegend(t.Data):
    enabled: bool
    path: str
    url: str
    template: t.ITemplate


#:export ILayer
class Layer(gws.Object, t.ILayer):
    @property
    def props(self):
        return types.LayerProps(
            extent=self.extent if self.extent != self.map.extent else None,
            meta=gws.common.metadata.props(self.meta),
            opacity=self.opacity,
            options=self.client_options,
            resolutions=self.resolutions,
            title=self.title,
            uid=self.uid,
            isClient=False,
        )

    @property
    def description(self) -> str:
        context = {'layer': self}
        return self.description_template.render(context).content

    @cached_property
    def has_search(self) -> bool:
        return len(self.get_children('gws.ext.search.provider')) > 0

    @property
    def has_legend(self) -> bool:
        return self.legend.enabled

    @cached_property
    def own_bounds(self) -> t.Optional[t.Bounds]:
        return

    @property
    def default_search_provider(self) -> t.Optional[t.ISearchProvider]:
        return

    @property
    def legend_url(self):
        return gws.SERVER_ENDPOINT + f'/cmd/mapHttpGetLegend/layerUid/{self.uid}'

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

        self.is_group: bool = False

        self.is_public: bool = self.root.application.auth.get_role('all').can_use(self)
        self.is_editable: bool = False

        self.legend: t.LayerLegend = t.LayerLegend(enabled=False)

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

        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'), gws.common.template.BUILTINS)
        self.description_template: t.ITemplate = gws.common.template.find(self.templates, subject='layer.description')

        p = self.var('dataModel')
        self.data_model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None

        self.resolutions: t.List[float] = gws.gis.zoom.resolutions_from_config(self.var('zoom'), self.map.resolutions)
        if not self.resolutions:
            raise gws.Error(f'no resolutions')

        # NB: the extent will be configured later on in map._configure_extent
        self.extent: t.Optional[t.Extent] = None

        self.opacity: float = self.var('opacity')
        self.client_options = self.var('clientOptions')

        self.geometry_type: t.Optional[t.GeometryType] = None

        p = self.var('style')
        self.style: t.IStyle = (
            gws.common.style.from_config(p) if p
            else gws.common.style.from_props(t.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES)))

        self.supports_wms: bool = False
        self.supports_wfs: bool = False

        self.ows_name: str = self.var('ows.name') or self.uid.split('.')[-1]
        self.ows_feature_name: str = self.var('ows.featureName') or self.ows_name
        self._ows_enabled: bool = self.var('ows.enabled')
        self._ows_enabled_services_uids: t.List[str] = self.var('ows.enabledServices.uids') or []
        self._ows_enabled_services_pattern: t.Regex = self.var('ows.enabledServices.pattern')

        self.crs: str = self.var('crs') or self.map.crs

        p = self.var('editDataModel')
        self.edit_data_model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None
        self.edit_options: t.Data = self.var('edit')

        p = self.var('editStyle')
        self.edit_style: t.Optional[t.IStyle] = gws.common.style.from_config(p) if p else None

    def post_configure(self):
        super().post_configure()

        try:
            self.configure_search()
        except Exception as e:
            gws.log.exception()

        legend = self.configure_legend()
        if legend:
            self.legend = legend
            self.legend.options = self.var('legend.options', default={})

    def configure_metadata(self, provider_meta=None) -> t.MetaData:
        """Load metadata from the config or from a provider, whichever comes first."""

        title = self.var('title')

        # use, in order 1) configured metadata, 2) provider meta, 3) dummy meta with title only
        m = self.var('meta')

        if not m and provider_meta:
            m = t.Config(provider_meta)

        if not m:
            if title:
                m = t.Config(title=title)
            elif self.var('uid'):
                m = t.Config(title=self.var('uid'))
            else:
                m = t.Config()

        if title:
            # title at the top level config overrides meta title
            m.title = title

        return gws.common.metadata.from_config(m)

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

    def render_box(self, rv: t.MapRenderView, extra_params=None):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, rv: t.MapRenderView, style: t.IStyle = None) -> str:
        return gws.tools.svg.as_xml(self.render_svg_tags(rv, style))

    def render_svg_tags(self, rv: t.MapRenderView, style: t.IStyle = None) -> t.List[t.Tag]:
        return []

    def configure_legend(self) -> t.LayerLegend:
        p = self.var('legend')
        if not p.enabled:
            return t.LayerLegend(enabled=False)

        if p.path:
            return t.LayerLegend(enabled=True, path=p.path)

        if p.url:
            return t.LayerLegend(enabled=True, url=p.url)

        if p.template:
            return t.LayerLegend(enabled=True, template=self.create_child('gws.ext.template', p.template))

    def render_legend(self, context=None) -> t.Optional[str]:
        """Render a legend and return the path to the legend image."""

        if self.legend.path:
            return self.legend.path

        cache_path = gws.LEGEND_CACHE_DIR + '/' + self.uid + '.png'

        if self.legend.url:
            try:
                r = gws.gis.ows.request.raw_get(self.legend.url)
            except gws.gis.ows.error.Error as e:
                gws.log.error(f'layer {self.uid!r}: legend download failed: {e!r}')
                self.legend.enabled = False
                return

            gws.write_file_b(cache_path, r.content)
            self.legend.path = cache_path
            return self.legend.path

        if self.legend.template:
            self.legend.template.render(context, out_path=cache_path, format='png')
            self.legend.path = cache_path
            return self.legend.path

        img = self.render_legend_image(context)
        if img:
            gws.write_file_b(cache_path, img)
            self.legend.path = cache_path
            return self.legend.path

        self.legend.enabled = False

    def render_legend_image(self, context=None) -> bytes:
        pass

    def render_html_legend(self, context=None) -> str:
        """Render a legend in the html format."""

        if self.legend.template:
            return self.legend.template.render(context).content

        path = self.render_legend(context)
        if path:
            return _DEFAULT_LEGEND_HTML.replace('{path}', path)

        return ''

    def get_features(self, bounds: t.Bounds, limit: int = 0) -> t.List[t.IFeature]:
        return []

    def ows_enabled(self, service: t.IOwsService) -> bool:
        if not self._ows_enabled:
            return False
        if self._ows_enabled_services_uids:
            return service.uid in self._ows_enabled_services_uids
        if self._ows_enabled_services_pattern:
            return re.search(self._ows_enabled_services_pattern, service.uid) is not None
        if service.type == 'wms' and self.supports_wms:
            return True
        if service.type == 'wfs' and self.supports_wfs:
            return True
        if self.layers:
            return any(la.ows_enabled(service) for la in self.layers)
        return False
