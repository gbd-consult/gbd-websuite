import gws
import gws.base.metadata
import gws.base.template
import gws.base.model
import gws.base.legend
import gws.base.search
import gws.gis.crs
import gws.gis.source
import gws.gis.zoom
import gws.lib.style
import gws.lib.svg
import gws.types as t

from . import lib


class Config(gws.ConfigWithAccess):
    """Layer configuration"""

    # dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    cacheEnabled: bool = True
    cache: lib.Cache = {}  # type:ignore #: cache configuration
    clientOptions: lib.ClientOptions = {}  # type:ignore #: options for the layer display in the client
    display: lib.DisplayMode = lib.DisplayMode.box  #: layer display mode
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    grid: lib.Grid = {}  # type:ignore #: grid configuration
    imageFormat: lib.ImageFormat = lib.ImageFormat.png8  #: image format
    legendEnabled: bool = True
    legend: t.Optional[gws.ext.config.legend]  #: legend configuration
    metadata: t.Optional[gws.base.metadata.Config]  #: layer metadata
    opacity: float = 1  #: layer opacity
    ows: bool = True  # layer is enabled for OWS services
    searchEnabled: bool = True
    search: gws.base.search.finder.collection.Config = {}  # type:ignore #: layer search configuration
    templates: t.Optional[t.List[gws.ext.config.template]]  #: client templates
    title: str = ''  #: layer title
    zoom: t.Optional[gws.gis.zoom.Config]  #: layer resolutions and scales


class CustomConfig(gws.ConfigWithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers this configuration applies to
    clientOptions: t.Optional[lib.ClientOptions]  # options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: t.Optional[lib.DisplayMode]  #: layer display mode
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: gws.base.legend.Config = {}  # type:ignore #: legend configuration
    metadata: t.Optional[gws.base.metadata.Config]  #: layer metadata
    opacity: t.Optional[float]  #: layer opacity
    ows: bool = True  # layer is enabled for OWS services
    search: gws.base.search.finder.collection.Config = {}  # type:ignore #: layer search configuration
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
    metadata: gws.base.metadata.Props
    opacity: t.Optional[float]
    clientOptions: lib.ClientOptions
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
    cache: t.Optional[lib.Cache]
    grid: lib.Grid
    clientOptions: lib.ClientOptions
    displayMode: lib.DisplayMode
    cTemplates: gws.base.template.collection.Object
    cFinders: gws.base.search.finder.collection.Object
    legend: t.Optional[gws.base.legend.Object]

    def configure(self):
        self.configure_base()
        self.configure_source()

        if not self.configure_metadata():
            self.metadata = gws.base.metadata.from_args(title=self.title)

        if not self.configure_resolutions():
            self.resolutions = self.var('defaultResolutions')

        if not self.configure_extent():
            pass

        if not self.configure_search():
            pass

        if not self.configure_legend():
            pass

    def set_metadata(self, *args):
        self.metadata = gws.base.metadata.from_dict(gws.to_dict(args[0]))
        self.metadata.extend(*args[1:])

    def configure_base(self):
        self.metadata = t.cast(gws.IMetadata, None)
        self.title = self.var('title')

        self.crs = self.var('defaultCrs')
        self.extent = t.cast(gws.Extent, None)
        self.imageFormat = self.var('imageFormat')
        self.opacity = self.var('opacity')
        self.resolutions = []

        self.cache = None
        if self.var('cacheEnabled'):
            self.cache = self.var('cache')
        self.grid = self.var('grid', default=lib.Grid())

        self.clientOptions = self.var('clientOptions')
        self.displayMode = self.var('display')
        self.layers = []

        self.cTemplates = self.create_child(gws.base.template.collection.Object, gws.Config(
            templates=self.var('templates'),
            defaults=_DEFAULT_TEMPLATES))

        self.cFinders = self.create_child(gws.base.search.finder.collection.Object)
        self.legend = None

    def configure_source(self):
        pass

    def configure_metadata(self):
        p = self.var('metadata')
        if p:
            self.metadata = gws.base.metadata.from_config(p)
            return True

    def configure_search(self):
        if not self.var('searchEnabled'):
            return True
        p = self.var('search.providers')
        if p:
            for cfg in p:
                self.cFinders.add_finder(cfg)
            return True

    def configure_extent(self):
        p = self.var('extent')
        if p:
            self.extent = gws.gis.extent.from_list(p)
            if not self.extent:
                raise gws.ConfigurationError(f'invalid extent {p!r} in layer={self.uid!r}')
            return True

    def configure_resolutions(self):
        p = self.var('zoom')
        if p:
            self.resolutions = gws.gis.zoom.resolutions_from_config(p)
            if not self.resolutions:
                raise gws.ConfigurationError(f'invalid zoom configuration in layer={self.uid!r}')
            return True

    def configure_legend(self):
        if not self.var('legendEnabled'):
            return True
        p = self.var('legend')
        if p:
            self.legend = self.create_child(gws.ext.object.legend, p)
            return True

    def post_configure(self):
        self.hasCache = self.cache is not None
        self.hasSearch = len(self.cFinders.items) > 0
        self.hasLegend = self.legend is not None

    def props(self, user):
        p = gws.Data(
            extent=self.extent,
            metadata=self.metadata,
            opacity=self.opacity,
            clientOptions=self.clientOptions,
            resolutions=self.resolutions,
            title=self.title,
            uid=self.uid,
            layers=self.layers,
        )

        if self.displayMode == lib.DisplayMode.tile:
            p.type = 'tile'
            p.url = lib.layer_url_path(self.uid, kind='tile')
            p.tileSize = self.grid.tileSize

        if self.displayMode == lib.DisplayMode.box:
            p.type = 'box'
            p.url = lib.layer_url_path(self.uid, kind='box')

        return p

    def render_legend(self, args=None) -> t.Optional[gws.LegendRenderOutput]:
        """Render a legend and return the path to the legend image."""

        if not self.legend:
            return None

        def _get():
            out = self.legend.render()
            return out

        if not args:
            return gws.get_server_global('legend_' + self.uid, _get)

        return self.legend.render(args)

    def render_description(self, args=None):
        tpl = self.cTemplates.find(subject='layer.description')
        if not tpl:
            return
        args = gws.merge({
            'layer': self,
            'service_metadata': gws.get(self, 'provider.metadata.values'),
            'source_layers': gws.get(self, 'source_layers'),
        }, args)
        return tpl.render(gws.TemplateRenderInput(args=args))

    def mapproxy_config(self, mc):
        pass

