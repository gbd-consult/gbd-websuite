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

from . import util


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    # dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    cache: t.Optional[util.CacheConfig]  # cache configuration
    clientOptions: util.ClientOptions = {}  # type:ignore #: options for the layer display in the client
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.box  #: layer display mode
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    grid: util.GridConfig = {}  # type:ignore #: grid configuration
    imageFormat: util.ImageFormat = util.ImageFormat.png8  #: image format
    legend: t.Optional[gws.ext.config.legend]  #: legend configuration
    metadata: t.Optional[gws.Metadata]  #: layer metadata
    opacity: float = 1  #: layer opacity
    ows: bool = True  # layer is enabled for OWS services
    search: t.Optional[util.SearchConfig] = {}  # type:ignore #: layer search configuration
    templates: t.Optional[t.List[gws.ext.config.template]]  #: client templates
    title: str = ''  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class CustomConfig(gws.ConfigWithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers this configuration applies to
    clientOptions: t.Optional[util.ClientOptions]  # options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: t.Optional[gws.LayerDisplayMode]  #: layer display mode
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: gws.base.legend.Config = {}  # type:ignore #: legend configuration
    metadata: t.Optional[gws.Metadata]  #: layer metadata
    opacity: t.Optional[float]  #: layer opacity
    ows: bool = True  # layer is enabled for OWS services
    # search: gws.base.search.finder.collection.Config = {}  # type:ignore #: layer search configuration
    templates: t.Optional[t.List[gws.ext.config.template]]  #: client templates
    title: t.Optional[str]  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class Props(gws.Props):
    model: t.Optional[gws.ext.props.model]
    editAccess: t.Optional[t.List[str]]
    # editStyle: t.Optional[gws.lib.style.Props]
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    metadata: gws.lib.metadata.Props
    opacity: t.Optional[float]
    clientOptions: util.ClientOptions
    resolutions: t.Optional[t.List[float]]
    # style: t.Optional[gws.lib.style.Props]
    tileSize: int = 0
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
    cache: t.Optional[util.Cache]
    grid: util.Grid
    clientOptions: util.ClientOptions

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

        self.layers = []

        self.cache = None
        if self.var('cache.enabled'):
            self.cache = self.var('cache')
        self.grid = self.var('grid', default=util.Grid())
        self.hasCache = self.cache is not None

    ##

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

        if self.displayMode == gws.LayerDisplayMode.tile:
            p.type = 'tile'
            p.url = self.url_path('tile')
            p.tileSize = self.grid.tileSize

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
