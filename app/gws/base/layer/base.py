import gws
import gws.base.model
import gws.base.model
import gws.base.search
import gws.base.template
import gws.gis.crs
import gws.gis.extent
import gws.gis.source
import gws.gis.zoom
import gws.gis.zoom
import gws.base.metadata
import gws.lib.style
import gws.lib.style
import gws.lib.svg

import gws.types as t


##


    # cache: CacheConfig
    # cache_uid: str
    # grid: GridConfig
    # grid_uid: str
    #
    # @property
    #
    # @property
    # def has_cache(self) -> bool:
    #     return self.cache.enabled
    #
    # @property
    # def has_search(self) -> bool:
    #     return len(self.search_providers) > 0
    #
    # @property
    # def has_legend(self) -> bool:
    #     return self.legend.enabled
    #
    # @property
    # def own_bounds(self) -> t.Optional[gws.Bounds]:
    #     return None
    #
    # @property
    # def default_search_provider(self) -> t.Optional[gws.IFinder]:
    #     return None
    #
    # @property
    # def legend_url(self):
    #     return layer_url_path(self.uid, kind='legend')
    #
    # @property
    # def ancestors(self):
    #     ps = []
    #     p = self.parent
    #     while p and p.is_a('gws.ext.layer'):
    #         ps.append(p)
    #         p = p.parent
    #     return ps
    #
    # def configure(self):
    #     self.map = self.get_closest('gws.base.map')
    #
    #
    #
    # def configure_source(self):
    #     pass
    #
    # def configure_metadata(self):
    #     p = self.var('metadata')
    #     if not p:
    #         return
    #
    #     # @TODO: implement metadata extension, e.g. "order=layer,project,provider"
    #
    #     self.set_metadata(p)
    #
    #
    #
    #
    # def post_configure(self):
    #     if not self.resolutions and self.map:
    #         self.resolutions = self.map.resolutions
    #     if not self.resolutions:
    #         raise gws.Error(f'no resolutions defined in layer={self.uid!r}')
    #
    #     if not self.metadata:
    #         self.set_metadata()
    #
    #     # title at the top level config preferred
    #     title = self.var('title') or self.metadata.get('title') or self.var('uid')
    #     self.metadata.set('title', title)
    #     self.title = title
    #
    # def edit_access(self, user):
    #     # @TODO granular edit access
    #
    #     if self.is_editable and self.edit_options and user.can_use(self.edit_options, parent=self):
    #         return ['all']
    #
    # # def edit_operation(self, operation: str, feature_props: t.List[gws.base.feature.Props]) -> t.List[gws.IFeature]:
    # #     pass
    # #
    #
    # def mapproxy_config(self, mc):
    #     pass
    #
    # def render_box(self, view, extra_params=None):
    #     return None
    #
    # def render_xyz(self, x, y, z):
    #     return None
    #
    # def render_svg_element(self, view, style=None):
    #     fr = self.render_svg_fragment(view, style)
    #     if fr:
    #         return gws.lib.svg.fragment_to_element(fr)
    #
    # def render_svg_fragment(self, view, style=None):
    #     return []
    #
    # def get_features(self, bounds: gws.Bounds, limit: int = 0) -> t.List[gws.IFeature]:
    #     return []
