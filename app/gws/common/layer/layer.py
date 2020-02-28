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
import gws.gis.svg
import gws.gis.zoom
import gws.tools.misc
import gws.tools.net

import gws.types as t
from gws import cached_property

from . import types

_DEFAULT_STYLE_VALUES = {
    'fill': 'rgba(0,0,0,1)',
    'stroke': 'rgba(0,0,0,1)',
    'stoke_width': 1,
}


class Config(t.WithTypeAndAccess):
    """Layer"""

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


#:export ILayer
class Layer(gws.Object, t.ILayer):
    def __init__(self):
        super().__init__()

        self.can_render_box = False
        self.can_render_xyz = False
        self.can_render_svg = False

        self.is_public = False
        self.is_editable = False

        self.has_cache = False
        self.has_legend = False

        self.supports_wms = False
        self.supports_wfs = False

        self.display = ''
        self.image_format = ''

        #:noexport
        self.cache: types.CacheConfig = None
        #:noexport
        self.grid: types.GridConfig = None

        self.cache_uid = None
        self.grid_uid = None

        self.layers = []

        self.map: t.IMap = None
        self.meta: t.MetaData = None

        self.description_template: t.ITemplate = None
        self.feature_format: t.IFormat = None
        self.data_model: t.IModel = None

        self.title = ''

        self.resolutions: t.List[float] = None
        self.extent: t.Extent = None

        self.legend_url = ''

        self.opacity = 1
        self.client_options = t

        self.services = []
        self.geometry_type = None

        self.style: t.IStyle = None
        self.edit_style: t.IStyle = None
        self.edit_data_model: t.IModel = None
        self.edit_options: t.Data = None

        self.ows_name = ''
        self.ows_services_enabled = []
        self.ows_services_disabled = []

    @property
    def props(self):
        return types.LayerProps({
            'extent': self.extent if self.extent != self.map.extent else None,
            'meta': self.meta,
            'opacity': self.opacity,
            'options': self.client_options,
            'resolutions': self.resolutions if self.resolutions != self.map.resolutions else None,
            'title': self.title,
            'uid': self.uid,
        })

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

        self.load_metadata()

        self.map = self.get_closest('gws.common.map')
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
            self.style = gws.common.style.from_config(self.root, p)
        else:
            self.style = gws.common.style.from_props(self.root, t.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES))

        p = self.var('edit')
        if p:
            self.edit_options = p

        self.cache = self.var('cache')
        self.has_cache = self.cache and self.cache.enabled

        self.grid = self.var('grid')

    def post_configure(self):
        super().post_configure()
        self._configure_search()

    def load_metadata(self, provider_meta=None):
        """Load metadata from the config or from a provider, whichever comes first."""

        title = self.var('title')

        # use, in order 1) configured metadata, 2) provider meta, 3) dummy meta with title only
        meta = self.var('meta') or provider_meta
        if not meta:
            if title:
                meta = t.MetaData(title=title)
            elif self.var('uid'):
                meta = t.MetaData(title=self.var('uid'))
            else:
                meta = t.MetaData()

        if title:
            # title at the top level config overrides meta title
            meta.title = title

        self.meta = gws.common.metadata.read(meta)
        self.title = self.meta.title

        uid = self.var('uid') or gws.as_uid(self.title) or 'layer'
        map = self.get_closest('gws.common.map')
        if map:
            uid = map.uid + '.' + uid
        self.set_uid(uid)

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

    def render_box(self, rv: t.RenderView, client_params=None):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, rv: t.RenderView, style: t.IStyle = None):
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

    def ows_enabled(self, service: t.IOwsService) -> bool:
        if service.type == 'wms' and not self.supports_wms:
            return False
        if service.type == 'wfs' and not self.supports_wfs:
            return False
        if self.ows_services_disabled and service.name in self.ows_services_disabled:
            return False
        if self.ows_services_enabled:
            return service.name in self.ows_services_enabled
        return True

    def _configure_search(self):
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
            self.add_child('gws.ext.search.provider', cfg)
