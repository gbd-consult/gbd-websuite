import re

import gws
import gws.base.metadata
import gws.base.model
import gws.base.search
import gws.base.style
import gws.base.template
import gws.lib.extent
import gws.lib.gis
import gws.lib.img
import gws.lib.legend
import gws.lib.ows
import gws.lib.svg
import gws.lib.units
import gws.lib.zoom
import gws.types as t
from . import types

_DEFAULT_STYLE = gws.Config(
    type='css',
    values={
        'fill': 'rgba(0,0,0,1)',
        'stroke': 'rgba(0,0,0,1)',
        'stoke_width': 1,
    }
)

_DEFAULT_LEGEND_HTML = """<div class="legend"><img src="{path}"/></div>"""


# layer urls, handled by the map action (base/map/action.py)

def url_for_get_box(layer_uid) -> str:
    return gws.action_url('mapGetBox', layerUid=layer_uid)


def url_for_get_tile(layer_uid, xyz=None) -> str:
    args = {
        'layerUid': layer_uid
    }
    if xyz:
        args['z'] = xyz.z
        args['x'] = xyz.x
        args['y'] = xyz.y
    return gws.action_url('mapGetXYZ', **args) + '/gws.png'


def url_for_get_legend(layer_uid) -> str:
    return gws.action_url('mapGetLegend', layerUid=layer_uid) + '/gws.png'


def url_for_get_features(layer_uid) -> str:
    return gws.action_url('mapGetFeatures', layerUid=layer_uid)


#

class Object(gws.Object, gws.ILayer):
    map: gws.IMap

    can_render_box: bool = False
    can_render_xyz: bool = False
    can_render_svg: bool = False

    is_group: bool = False
    is_public: bool = False
    is_editable: bool = False

    supports_wms: bool = False
    supports_wfs: bool = False

    cache: types.CacheConfig
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
    grid: types.GridConfig
    grid_uid: str
    image_format: str
    layers: t.List[gws.ILayer]
    legend: gws.Legend
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

    has_configured_extent = False
    has_configured_layers = False
    has_configured_legend = False
    has_configured_metadata = False
    has_configured_resolutions = False
    has_configured_search = False

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
        return url_for_get_legend(self.uid)

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

        uid = self.var('uid') or gws.as_uid(self.var('title'))
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

        self.is_public = self.root.application.auth.get_role('all').can_use(self)
        self.cache = self.var('cache', default=types.CacheConfig(enabled=False))
        self.cache_uid = ''
        self.client_options = self.var('clientOptions')
        self.crs = self.var('crs') or (self.map.crs if self.map else gws.EPSG_3857)
        self.display = self.var('display')
        self.edit_options = self.var('edit')
        self.geometry_type = None
        self.grid = self.var('grid', default=types.GridConfig())
        self.grid_uid = ''
        self.image_format = self.var('imageFormat')
        self.layers = []
        self.legend = gws.Legend(enabled=False)
        self.metadata = t.cast(gws.IMetaData, None)
        self.opacity = self.var('opacity')
        self.ows_enabled = self.var('ows.enabled')
        self.ows_enabled_services_uids = self.var('ows.enabledServices.uids') or []
        self.ows_enabled_services_pattern = self.var('ows.enabledServices.pattern')
        self.ows_feature_name = ''
        self.ows_name = ''
        self.resolutions = []
        self.search_providers = []

        self.data_model = self.create_child_if_config(gws.base.model.Object, self.var('dataModel'))
        self.edit_data_model = self.create_child_if_config(gws.base.model.Object, self.var('editDataModel'))

        self.templates = gws.base.template.bundle.create(
            self.root,
            gws.Config(templates=self.var('templates'), withBuiltins=True),
            parent=self)
        self.description_template = self.templates.find(subject='layer.description')

        self.style = t.cast(gws.IStyle, self.create_child(gws.base.style.Object, self.var('style') or _DEFAULT_STYLE))
        self.edit_style = self.create_child_if_config(gws.base.style.Object, self.var('editStyle'))

        p = self.var('metaData')
        if p:
            self.configure_metadata_from(p)
            self.has_configured_metadata = True

        p = self.var('extent')
        if p:
            self.extent = gws.lib.extent.from_list(p)
            if not self.extent:
                raise gws.Error(f'invalid extent {p!r} in layer={self.uid!r}')
            self.has_configured_extent = True

        p = self.var('zoom')
        if p:
            self.resolutions = gws.lib.zoom.resolutions_from_config(p, self.map.resolutions if self.map else [])
            if not self.resolutions:
                raise gws.Error(f'invalid zoom configuration in layer={self.uid!r}')
            self.has_configured_resolutions = True

        p = self.var('search')
        if p:
            if not p.enabled:
                self.search_providers = []
                self.has_configured_search = True
            elif p.providers:
                self.search_providers = [
                    t.cast(gws.ISearchProvider, self.create_child('gws.ext.search.provider', c))
                    for c in p.providers]
                self.has_configured_search = True

        p = self.var('legend')
        if p:
            if not p.enabled:
                self.legend = gws.Legend(enabled=False)
                self.has_configured_legend = True
            elif p.path:
                self.legend = gws.Legend(enabled=True, path=p.path, options=p.options or {})
                self.has_configured_legend = True
            elif p.url:
                self.legend = gws.Legend(enabled=True, urls=[p.url], cache_max_age=p.cacheMaxAge or 0, options=p.options or {})
                self.has_configured_legend = True
            elif p.template:
                tpl = self.create_child('gws.ext.template', p.template)
                self.legend = gws.Legend(enabled=True, template=tpl, options=p.options or {})
                self.has_configured_legend = True

    def configure_metadata_from(self, m: gws.base.metadata.Record):
        self.metadata = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, m))
        self.title = self.var('title') or self.metadata.title
        self.ows_name = self.var('ows.name') or self.uid.split('.')[-1]
        self.ows_feature_name = self.var('ows.featureName') or self.ows_name
        self.has_configured_metadata = True

    def post_configure(self):
        if not self.resolutions:
            if self.map:
                self.resolutions = self.map.resolutions

        if not self.resolutions:
            raise gws.Error(f'no resolutions defined in layer={self.uid!r}')

        if not self.metadata:
            self.configure_metadata_from(gws.base.metadata.Record(
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
            return gws.get_server_global('legend_' + self.uid, _get)

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
        if service.protocol == gws.OwsProtocol.WMS and self.supports_wms:
            return True
        if service.protocol == gws.OwsProtocol.WFS and self.supports_wfs:
            return True
        if self.layers:
            return any(la.enabled_for_ows(service) for la in self.layers)
        return False
