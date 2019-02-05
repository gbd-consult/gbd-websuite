import gws
import gws.common.format
import gws.common.search
import gws.common.template
import gws.gis.cache
import gws.gis.feature
import gws.gis.mpx as mpx
import gws.gis.proj
import gws.gis.source
import gws.gis.zoom
import gws.types as t

import gws.tools.net


class ClientOptions(t.Data):
    """client options for a layer"""

    expanded: t.Optional[bool] = False  #: the layer is expanded in the list view
    listed: t.Optional[bool] = True  #: the layer is displayed in this list view
    selected: t.Optional[bool] = False  #: the layer is intially selected
    visible: t.Optional[bool] = True  #: the layer is intially visible
    unfolded: t.Optional[bool] = False  #: the layer is not listed, but its children are
    exclusive: t.Optional[bool] = False  #: only one of this layer's children is visible at a time


_default_client_opts = ClientOptions({
    'visible': True,
    'expanded': False,
    'listed': True,
    'selected': False,
})


class BaseConfig(t.WithTypeAndAccess):
    """map layer"""

    cache: t.Optional[t.CacheConfig]  #: cache configuration
    clientOptions: t.Optional[ClientOptions]  #: options for the layer display in the client
    description: t.Optional[t.TemplateConfig]  #: template for the layer description
    editable: t.Optional[bool] = False  #: this layer is editable
    extent: t.Optional[t.Extent]  #: layer extent
    featureFormat: t.Optional[t.FormatConfig]  #: feature formatting options
    grid: t.Optional[t.GridConfig]  #: grid configuration
    legend: t.Optional[t.url]  #: legend url
    meta: t.Optional[t.MetaConfig]  #: layer meta data
    opacity: t.Optional[float]  #: layer opacity
    search: t.Optional[gws.common.search.Config]  #: layer search configuration
    source: t.Optional[t.ext.gis.source.Config]  #: data source for the layer
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    title: t.Optional[str]  #: layer title
    uid: t.Optional[str]  #: layer unique id
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class ProxiedConfig(BaseConfig):
    pass


class BaseProps(t.Data):
    description: str = ''
    editable: t.Optional[bool]
    extent: t.Optional[t.Extent]
    meta: t.MetaData
    opacity: t.Optional[float]
    options: ClientOptions
    resolutions: t.Optional[t.List[float]]
    title: str
    type: str
    uid: str


class Base(gws.PublicObject, t.LayerObject):
    def __init__(self):
        super().__init__()

        self.cache = None
        self.grid = None
        self.map = None
        self.meta = None
        self.opacity = None
        self.source = None

        self.description_template = None
        self.feature_format = None

        self.template = None
        self.title = None

        self.resolutions = []
        self.extent = []
        self.crs = ''

        self.legend = ''
        self.has_no_cache_variant = False

    def configure(self):
        super().configure()

        if self.var('source'):
            self.source = self.add_child('gws.ext.gis.source', self.var('source'))

        self.meta = self.configure_meta()
        self.uid = self.var('uid') or gws.as_uid(self.title)

        self.map = self.get_closest('gws.common.map')
        if self.map:
            self.uid = self.map.uid + '.' + self.uid

        self.cache = self.var('cache', parent=True)
        self.grid = self.var('grid', parent=True)
        self.legend = self.var('legend')
        self.opacity = self.var('opacity', parent=True)

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

        self.resolutions = gws.gis.zoom.effective_resolutions(
            self.var('zoom'),
            t.cast(t.MapView, self.parent).resolutions)

        self.crs = self.map.crs
        self.extent = self.configure_extent()

        for p in self.var('search.providers', default=[]):
            self.add_child('gws.ext.search.provider', p)

    def configure_meta(self):
        m = self.var('meta') or t.MetaData()
        # title at the top level config preferred
        if self.var('title'):
            m.title = self.var('title')
        self.title = m.title
        return m

    def configure_extent(self):
        e = self.var('extent')
        if e:
            return e

        if self.source and self.source.extent:
            return gws.gis.proj.transform_bbox(
                self.source.extent,
                self.source.crs,
                self.crs
            )
        return t.cast(t.MapView, self.parent).extent

    @property
    def props(self):

        return gws.compact({
            'editable': False,
            'meta': self.meta,
            'opacity': self.opacity,
            'options': self.var('clientOptions', default=_default_client_opts),
            # @TODO: dont write those if equal to parent
            'extent': self.extent,
            'resolutions': self.resolutions,
            'title': self.title,
            'type': self.klass.split('.')[-1],
            'uid': self.uid,
            'description': self.description(),
        })

    def description(self, options=None):
        ctx = gws.defaults(options, {
            'layer': self,
            'service': self.source.service_metadata() if self.source else ''
        })
        return self.description_template.render(ctx).content

    def props_for(self, user):
        p = super().props_for(user)
        if p:
            p['editable'] = self.var('editable') and user.can('write', self)
        return p

    def mapproxy_config(self, mc, options=None):
        pass

    def render_bbox(self, bbox, width, height, **client_params):
        return None

    def render_xyz(self, x, y, z):
        return None

    def render_svg(self, bbox, dpi, scale, rotation, style):
        return None

    def get_features(self, bbox):
        return []

    def modify_features(self, operation, feature_params):
        pass


class Proxied(Base):
    def render_bbox(self, bbox, width, height, **client_params):
        forward = {}
        uid = self.uid
        if 'dpi' in client_params:
            forward['DPI__gws'] = client_params['dpi']
            if self.has_no_cache_variant:
                uid += '_NOCACHE'
        return mpx.wms_request(
            uid,
            bbox,
            width,
            height,
            self.crs,
            forward)

    def render_xyz(self, x, y, z):
        return mpx.wmts_request(self.uid, x, y, z, self.grid.tileSize)

    def mapproxy_config(self, mc, options=None):
        source = gws.get(options, 'source') or self.source.mapproxy_config(mc)

        # configure the "destination" grid and cache for this source
        # (which can be a raw source or a source cache)

        res = self.resolutions  # [:self.cache.maxLevel] ??? doesn't work
        if len(res) < 2:
            res = [self.resolutions[0], self.resolutions[0]]

        dst_grid_config = gws.compact({
            'origin': self.grid.origin,
            'tile_size': [self.grid.tileSize, self.grid.tileSize],
            'res': res,
            'srs': self.crs,
            'bbox': self.extent,
        })
        dst_grid_config = gws.extend(dst_grid_config, self.grid.options)
        dst_grid = mc.grid(self, dst_grid_config, self.uid + '_dst')

        dst_cache_options = {
            'type': 'file',
            'directory_layout': 'mp'
        }
        dst_cache_options = gws.extend(dst_cache_options, self.cache.options)

        dst_cache_config = {
            'sources': [source],
            'grids': [dst_grid],
            'cache': dst_cache_options,
            'meta_size': [self.grid.metaSize, self.grid.metaSize],
            'meta_buffer': self.grid.metaBuffer,
            'disable_storage': not self.cache.enabled,
            'minimize_meta_requests': True,
        }

        dst_cache = mc.cache(self, dst_cache_config, self.uid + '_dst')

        return mc.layer(self, {
            'title': self.uid,
            'sources': [dst_cache]
        })
