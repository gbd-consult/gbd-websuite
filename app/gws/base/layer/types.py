import gws
import gws.base.model
import gws.base.search
import gws.lib.gis.source
import gws.lib.metadata
import gws.lib.style
import gws.lib.gis.zoom
import gws.types as t


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False  #: the layer is expanded in the list view
    listed: t.Optional[bool] = True  #: the layer is displayed in this list view
    selected: t.Optional[bool] = False  #: the layer is intially selected
    visible: t.Optional[bool] = True  #: the layer is intially visible
    unfolded: t.Optional[bool] = False  #: the layer is not listed, but its children are
    exclusive: t.Optional[bool] = False  #: only one of this layer's children is visible at a time


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'  #: png 8-bit
    png24 = 'png24'  #: png 24-bit


class DisplayMode(t.Enum):
    """Layer display mode"""

    box = 'box'  #: display a layer as one big image (WMS-alike)
    tile = 'tile'  #: display a layer in a tile grid
    client = 'client'  #: draw a layer in the client


class CacheConfig(gws.Config):
    """Cache configuration"""

    enabled: bool = False  #: cache is enabled
    maxAge: gws.Duration = '7d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache


class GridConfig(gws.Config):
    """Grid configuration for caches and tiled data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    reqSize: int = 0  #: number of metatiles to fetch
    reqBuffer: int = 0  #: pixel buffer


class EditConfig(gws.WithAccess):
    """Edit access for a layer"""

    pass


class LegendConfig(gws.Config):
    """Layer legend confuguration."""

    cacheMaxAge: gws.Duration = '1d'  #: max cache age for external legends
    enabled: bool = True  #: the legend is enabled
    options: t.Optional[dict]  #: provider-dependent legend options
    path: t.Optional[gws.FilePath]  #: path of the legend image
    template: t.Optional[gws.ext.template.Config]  #: template for an HTML legend
    url: t.Optional[gws.Url]  #: url of the legend image


class Config(gws.WithAccess):
    """Layer configuration"""

    clientOptions: ClientOptions = {}  # type:ignore #: options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: DisplayMode = DisplayMode.box  #: layer display mode
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: LegendConfig = {}  # type:ignore #: legend configuration
    metadata: t.Optional[gws.lib.metadata.Config]  #: layer metadata
    opacity: float = 1  #: layer opacity
    ows: bool = True  # layer is enabled for OWS services
    search: gws.base.search.Config = {}  # type:ignore #: layer search configuration
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: client templates
    title: str = ''  #: layer title
    zoom: t.Optional[gws.lib.gis.zoom.Config]  #: layer resolutions and scales


class CustomConfig(gws.WithAccess):
    """Custom layer configuration"""

    applyTo: t.Optional[gws.lib.gis.source.LayerFilterConfig]  #: source layers this configuration applies to
    clientOptions: t.Optional[ClientOptions]  # options for the layer display in the client
    dataModel: t.Optional[gws.base.model.Config]  #: layer data model
    display: t.Optional[DisplayMode]  #: layer display mode
    edit: t.Optional[EditConfig]  #: editing permissions
    extent: t.Optional[gws.Extent]  #: layer extent
    extentBuffer: t.Optional[int]  #: extent buffer
    legend: t.Optional[LegendConfig]  # #: legend configuration
    metadata: t.Optional[gws.lib.metadata.Config]  #: layer metadata
    opacity: t.Optional[float]  #: layer opacity
    ows: bool = True  # layer is enabled for OWS services
    search: t.Optional[gws.base.search.Config]  #: layer search configuration
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: client templates
    title: t.Optional[str]  #: layer title
    zoom: t.Optional[gws.lib.gis.zoom.Config]  #: layer resolutions and scales


class Props(gws.Props):
    dataModel: t.Optional[gws.base.model.Props]
    editAccess: t.Optional[t.List[str]]
    editStyle: t.Optional[gws.lib.style.Props]
    extent: t.Optional[gws.Extent]
    geometryType: t.Optional[gws.GeometryType]
    layers: t.Optional[t.List['Props']]
    loadingStrategy: t.Optional[str]
    metadata: gws.lib.metadata.Props
    opacity: t.Optional[float]
    options: ClientOptions
    resolutions: t.Optional[t.List[float]]
    style: t.Optional[gws.lib.style.Props]
    tileSize: int = 0
    title: str = ''
    type: str
    uid: str
    url: str = ''
