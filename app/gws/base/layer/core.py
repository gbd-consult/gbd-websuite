import gws
import gws.lib.metadata
import gws.base.template
import gws.base.model
import gws.base.legend
import gws.base.search
import gws.gis.crs
import gws.gis.bounds
import gws.gis.extent
import gws.gis.source
import gws.gis.zoom
import gws.lib.style
import gws.lib.svg
import gws.types as t


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'
    """png 8-bit"""
    png24 = 'png24'
    """png 24-bit"""


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False
    """the layer is expanded in the list view"""
    listed: t.Optional[bool] = True
    """the layer is displayed in this list view"""
    selected: t.Optional[bool] = False
    """the layer is intially selected"""
    visible: t.Optional[bool] = True
    """the layer is intially visible"""
    unfolded: t.Optional[bool] = False
    """the layer is not listed, but its children are"""
    exclusive: t.Optional[bool] = False
    """only one of this layer's children is visible at a time"""


class CacheConfig(gws.Config):
    """Cache configuration"""

    enabled: bool = True
    maxAge: gws.Duration = '7d'
    """cache max. age"""
    maxLevel: int = 1
    """max. zoom level to cache"""
    requestBuffer: t.Optional[int]
    requestTiles: t.Optional[int]


class Cache(gws.Data):
    maxAge: int
    maxLevel: int
    requestBuffer: int
    requestTiles: int


class GridConfig(gws.Config):
    """Grid configuration for caches and tiled data"""

    crs: t.Optional[gws.CrsName]
    extent: t.Optional[gws.Extent]
    corner: t.Optional[gws.Corner]
    resolutions: t.Optional[t.List[float]]
    tileSize: t.Optional[int]


class EditConfig(gws.ConfigWithAccess):
    """Edit access for a layer"""

    pass


class SearchConfig(gws.Config):
    enabled: bool = True
    """search is enabled"""
    finders: t.Optional[t.List[gws.ext.config.finder]]
    """search prodivers"""


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    models: t.Optional[t.List[gws.ext.config.model]]
    """data models"""
    cache: t.Optional[CacheConfig]
    """cache configuration"""
    clientOptions: ClientOptions = {}
    """options for the layer display in the client"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.box
    """layer display mode"""
    extent: t.Optional[gws.Extent]
    """layer extent"""
    extentBuffer: t.Optional[int]
    """extent buffer"""
    sourceGrid: t.Optional[GridConfig]
    """source grid"""
    targetGrid: t.Optional[GridConfig]
    """target (client) grid"""
    imageFormat: ImageFormat = ImageFormat.png8
    """image format"""
    legend: t.Optional[gws.ext.config.legend]
    """legend configuration"""
    metadata: t.Optional[gws.Metadata]
    """layer metadata"""
    opacity: float = 1
    """layer opacity"""
    ows: bool = True  # layer is enabled for OWS services
    search: t.Optional[SearchConfig] = {}  # type:ignore
    """layer search configuration"""
    templates: t.Optional[t.List[gws.ext.config.template]]
    """client templates"""
    title: str = ''
    """layer title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """layer resolutions and scales"""


class CustomConfig(gws.ConfigWithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.gis.source.LayerFilterConfig]
    """source layers this configuration applies to"""
    clientOptions: t.Optional[ClientOptions]  # options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]
    """layer data model"""
    display: t.Optional[gws.LayerDisplayMode]
    """layer display mode"""
    extent: t.Optional[gws.Extent]
    """layer extent"""
    extentBuffer: t.Optional[int]
    """extent buffer"""
    legend: gws.base.legend.Config = {}  # type:ignore
    """legend configuration"""
    metadata: t.Optional[gws.Metadata]
    """layer metadata"""
    opacity: t.Optional[float]
    """layer opacity"""
    ows: bool = True  # layer is enabled for OWS services
    # search: gws.base.search.finder.collection.Config = {}  # type:ignore
    """layer search configuration"""
    templates: t.Optional[t.List[gws.ext.config.template]]
    """client templates"""
    title: t.Optional[str]
    """layer title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """layer resolutions and scales"""


class GridProps(gws.Props):
    corner: str
    extent: gws.Extent
    resolutions: t.List[float]
    tileSize: int


class Props(gws.Props):
    model: t.Optional[gws.base.model.Props]
    editAccess: t.Optional[t.List[str]]
    # editStyle: t.Optional[gws.lib.style.Props]
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    metadata: gws.lib.metadata.Props
    opacity: t.Optional[float]
    clientOptions: ClientOptions
    resolutions: t.Optional[t.List[float]]
    # style: t.Optional[gws.lib.style.Props]
    grid: GridProps
    title: str = ''
    type: str
    uid: str
    url: str = ''


_DEFAULT_STYLE = gws.Config(
    values={
        'fill': 'rgba(0,0,0,1)',
        'stroke': 'rgba(0,0,0,1)',
        'stroke-width': 1,
    }
)

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/layer_description.cx.html',
        subject='layer.description',
        access=gws.PUBLIC,
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_description.cx.html',
        subject='feature.description',
        access=gws.PUBLIC,
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_teaser.cx.html',
        subject='feature.teaser',
        access=gws.PUBLIC,
    ),
]


class Object(gws.Node, gws.ILayer):
    cache: t.Optional[Cache]
    clientOptions: ClientOptions

    canRenderBox = False
    canRenderXyz = False
    canRenderSvg = False

    supportsRasterServices = False
    supportsVectorServices = False

    parentBounds: gws.Bounds
    parentResolutions: t.List[float]

    def configure(self):
        self.parentBounds = self.var('_parentBounds')
        self.parentResolutions = self.var('_parentResolutions')

        self.bounds = self.parentBounds
        self.clientOptions = self.var('clientOptions')
        self.displayMode = self.var('display')
        self.imageFormat = self.var('imageFormat')
        self.opacity = self.var('opacity')
        self.resolutions = self.parentResolutions
        self.title = self.var('title')

        self.metadata = gws.Metadata()
        self.legend = None

        self.templateMgr = self.create_child(gws.base.template.manager.Object, gws.Config(
            templates=self.var('templates'),
            defaults=_DEFAULT_TEMPLATES))

        self.searchMgr = self.create_child(gws.base.search.manager.Object)

        p = self.var('models')
        self.modelMgr = self.create_child(gws.base.model.manager.Object, gws.Config(
            models=p or [gws.Config(type='default')]))

        self.layers = []

        self.sourceGrid = None
        self.targetGrid = None

        p = self.var('targetGrid')
        if p and p.crs and p.crs != self.parentBounds.crs:
            raise gws.Error(f'invalid target grid crs')

        self.cache = None
        if self.var('cache.enabled'):
            self.cache = Cache(self.var('cache'))

    ##

    def ancestors(self):
        ls = []
        p = self.parent
        while isinstance(p, Object):
            ls.append(p)
            p = p.parent
        return ls

    _url_path_suffix = '/gws.png'

    def url_path(self, kind):
        # layer urls, handled by the map action (base/map/action.py)
        if kind == 'box':
            return gws.action_url_path('mapGetBox', layerUid=self.uid) + self._url_path_suffix
        if kind == 'tile':
            return gws.action_url_path('mapGetXYZ', layerUid=self.uid) + '/z/{z}/x/{x}/y/{y}' + self._url_path_suffix
        if kind == 'legend':
            return gws.action_url_path('mapGetLegend', layerUid=self.uid) + self._url_path_suffix
        if kind == 'features':
            return gws.action_url_path('mapGetFeatures', layerUid=self.uid)

    def props(self, user):
        p = gws.Data(
            extent=self.bounds.extent,
            metadata=gws.lib.metadata.props(self.metadata),
            opacity=self.opacity,
            clientOptions=self.clientOptions,
            resolutions=sorted(self.resolutions, reverse=True),
            title=self.title,
            uid=self.uid,
            layers=self.layers,
        )

        if self.targetGrid:
            p.grid = GridProps(
                corner=self.targetGrid.corner,
                extent=self.targetGrid.bounds.extent,
                resolutions=sorted(self.targetGrid.resolutions, reverse=True),
                tileSize=self.targetGrid.tileSize,
            )

        if self.displayMode == gws.LayerDisplayMode.tile:
            p.type = 'tile'
            p.url = self.url_path('tile')

        if self.displayMode == gws.LayerDisplayMode.box:
            p.type = 'box'
            p.url = self.url_path('box')

        return p

    def render_legend(self, args=None) -> t.Optional[gws.LegendRenderOutput]:

        if not self.legend:
            return None

        def _get():
            out = self.legend.render()
            return out

        if not args:
            return gws.get_server_global('legend_' + self.uid, _get)

        return self.legend.render(args)

    def mapproxy_config(self, mc):
        pass
