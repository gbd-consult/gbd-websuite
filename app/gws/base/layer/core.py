import re

import gws
import gws.base.model
import gws.base.template
import gws.lib.crs
import gws.lib.extent
import gws.lib.gis.zoom
import gws.lib.legend
import gws.lib.metadata
import gws.lib.style
import gws.lib.svg
import gws.types as t

from . import types

_DEFAULT_STYLE = gws.Config(
    values={
        'fill': 'rgba(0,0,0,1)',
        'stroke': 'rgba(0,0,0,1)',
        'stoke_width': 1,
    }
)

# layer urls, handled by the map action (base/map/action.py)

_suffix = '/gws.png'


def layer_url_path(layer_uid, kind: t.Literal['box', 'tile', 'legend', 'features']) -> str:
    if kind == 'box':
        return gws.action_url_path('mapGetBox', layerUid=layer_uid) + _suffix
    if kind == 'tile':
        return gws.action_url_path('mapGetXYZ', layerUid=layer_uid) + '/z/{z}/x/{x}/y/{y}' + _suffix
    if kind == 'legend':
        return gws.action_url_path('mapGetLegend', layerUid=layer_uid) + _suffix
    if kind == 'features':
        return gws.action_url_path('mapGetFeatures', layerUid=layer_uid)


##


class Object(gws.Node, gws.ILayer):
    cache: types.CacheConfig
    cache_uid: str
    grid: types.GridConfig
    grid_uid: str
    description_template: t.Optional[gws.ITemplate]

    @property
    def description(self) -> str:
        if not self.description_template:
            return ''
        context = {'layer': self}
        return gws.to_str(self.description_template.render(context).content)

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
        return layer_url_path(self.uid, kind='legend')

    @property
    def ancestors(self):
        ps = []
        p = self.parent
        while p and p.is_a('gws.ext.layer'):
            ps.append(p)
            p = p.parent
        return ps

    def configure(self):
        self.map = self.get_closest('gws.base.map')

        uid = self.var('uid') or gws.to_uid(self.var('title'))
        if self.map:
            uid = self.map.uid + '.' + uid
        self.set_uid(uid)

        p = self.var('crs')
        self.crs = gws.lib.crs.require(p) if p else (
            self.map.crs if self.map else gws.lib.crs.get3857())

        self.cache = self.var('cache', default=types.CacheConfig(enabled=False))
        self.cache_uid = ''
        self.client_options = self.var('clientOptions')
        self.display = self.var('display')
        self.edit_options = self.var('edit')
        self.grid = self.var('grid', default=types.GridConfig())
        self.grid_uid = ''
        self.image_format = self.var('imageFormat')
        self.layers = []
        self.legend = gws.Legend(enabled=False)
        self.opacity = self.var('opacity')
        self.ows_enabled = self.var('ows')

        self.resolutions = []
        self.search_providers = []

        self.data_model = self.create_child_if_config(gws.base.model.Object, self.var('dataModel'))
        self.edit_data_model = self.create_child_if_config(gws.base.model.Object, self.var('editDataModel'))

        self.templates = gws.base.template.bundle.create(
            self.root,
            gws.Config(templates=self.var('templates'), withBuiltins=True),
            parent=self)
        self.description_template = self.templates.find(subject='layer.description')

        self.style = gws.lib.style.from_config(self.var('style', default=_DEFAULT_STYLE))
        p = self.var('editStyle')
        if p:
            self.edit_style = gws.lib.style.from_config(p)

        self.configure_source()
        self.configure_metadata()
        self.configure_extent()
        self.configure_zoom()
        self.configure_search()
        self.configure_legend()

    def configure_source(self):
        pass

    def configure_metadata(self):
        p = self.var('metadata')
        if not p:
            return

        if not p.get('extend'):
            self.set_metadata(p)
            return True

        if p.get('extend') == gws.lib.metadata.ExtendOption.app:
            project: gws.IProject = self.get_closest('gws.base.project')
            self.set_metadata(p, project.metadata)
            return True

    def configure_extent(self):
        p = self.var('extent')
        if not p:
            return
        self.extent = gws.lib.extent.from_list(p)
        if not self.extent:
            raise gws.Error(f'invalid extent {p!r} in layer={self.uid!r}')
        return True

    def configure_zoom(self):
        p = self.var('zoom')
        if not p:
            return
        self.resolutions = gws.lib.gis.zoom.resolutions_from_config(
            p, self.map.resolutions if self.map else [])
        if not self.resolutions:
            raise gws.Error(f'invalid zoom configuration in layer={self.uid!r}')
        return True

    def configure_search(self):
        p = self.var('search')
        if not p:
            return

        if not p.enabled:
            self.search_providers = []
            return True

        if p.providers:
            self.search_providers = self.create_children('gws.ext.search.provider', p.providers)
            return True

    def configure_legend(self):
        p = self.var('legend')
        if not p:
            return

        if not p.enabled:
            self.legend = gws.Legend(enabled=False)
            return True

        if p.path:
            self.legend = gws.Legend(enabled=True, path=p.path, options=p.options or {})
            return True

        if p.url:
            self.legend = gws.Legend(enabled=True, urls=[p.url], cache_max_age=p.cacheMaxAge or 0, options=p.options or {})
            return True

        if p.template:
            tpl = self.require_child('gws.ext.template', p.template)
            self.legend = gws.Legend(enabled=True, template=tpl, options=p.options or {})
            return True

    def set_metadata(self, *args):
        self.metadata = gws.lib.metadata.from_args(title=self.var('title'))
        for a in args:
            self.metadata.extend(a)
        self.title = self.metadata.get('title')

    def post_configure(self):
        if not self.resolutions and self.map:
            self.resolutions = self.map.resolutions
        if not self.resolutions:
            raise gws.Error(f'no resolutions defined in layer={self.uid!r}')

        if not self.metadata:
            self.set_metadata(gws.lib.metadata.from_args(
                title=self.var('title') or self.var('uid') or 'layer'
            ))

    def edit_access(self, user):
        # @TODO granular edit access

        if self.is_editable and self.edit_options and user.can_use(self.edit_options, parent=self):
            return ['all']

    # def edit_operation(self, operation: str, feature_props: t.List[gws.lib.feature.Props]) -> t.List[gws.IFeature]:
    #     pass
    #
    def props_for(self, user):
        return types.Props(
            extent=self.extent,
            metadata=self.metadata,
            editAccess=self.edit_access(user),
            opacity=self.opacity,
            options=self.client_options,
            resolutions=self.resolutions,
            title=self.title,
            uid=self.uid,
        )

    def mapproxy_config(self, mc):
        pass

    def render_box(self, rv: gws.MapRenderView, extra_params=None):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> str:
        return gws.lib.svg.to_xml(self.render_svg_tags(rv, style))

    def render_svg_tags(self, rv: gws.MapRenderView, style: gws.IStyle = None) -> t.List[gws.Tag]:
        return []

    def render_legend_with_cache(self, context=None) -> t.Optional[gws.LegendRenderOutput]:
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
