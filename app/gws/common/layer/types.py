import gws.types as t


class ImageFormat(t.Enum):
    """Image format"""

    png8 = 'png8'  #: png 8-bit
    png24 = 'png24'  #: png 24-bit


class DisplayMode(t.Enum):
    """Layer display mode"""

    box = 'box'  #: display a layer as one big image (WMS-alike)
    tile = 'tile'  #: display a layer in a tile grid
    client = 'client'  #: draw a layer in the client


class CacheConfig(t.Config):
    """Cache configuration"""

    enabled: bool = False  #: cache is enabled
    maxAge: t.Duration = '7d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache


class GridConfig(t.Config):
    """Grid configuration for caches and tiled data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    metaSize: int = 0  #: number of meta-tiles to fetch
    metaBuffer: int = 0  #: pixel buffer


class ClientOptions(t.Data):
    """Client options for a layer"""

    expanded: t.Optional[bool] = False  #: the layer is expanded in the list view
    listed: t.Optional[bool] = True  #: the layer is displayed in this list view
    selected: t.Optional[bool] = False  #: the layer is intially selected
    visible: t.Optional[bool] = True  #: the layer is intially visible
    unfolded: t.Optional[bool] = False  #: the layer is not listed, but its children are
    exclusive: t.Optional[bool] = False  #: only one of this layer's children is visible at a time


class EditConfig(t.WithAccess):
    pass


class LegendConfig(t.Config):
    enabled: bool = True
    url: t.Optional[t.Url]


class FlattenConfig(t.Config):
    """Flatten the layer hierarchy."""

    level: int  #: flatten level
    useGroups: bool = False  #: use group names (true) or image layer names (false)


class OwsConfig(t.Config):
    name: str = ''  #: layer name for ows services
    servicesEnabled: t.Optional[t.List[str]]  #: services enabled for this layer
    servicesDisabled: t.Optional[t.List[str]]  #: services disabled for this layer


class LayerProps(t.Data):
    editAccess: t.Optional[t.List[str]]
    editStyle: t.Optional[t.StyleProps]
    extent: t.Optional[t.Extent]
    geometryType: str = ''
    layers: t.Optional[t.List['LayerProps']]
    loadingStrategy: t.Optional[str]
    meta: t.MetaData
    opacity: t.Optional[float]
    options: ClientOptions
    resolutions: t.Optional[t.List[float]]
    style: t.Optional[t.StyleProps]
    tileSize: int = 0
    title: str = ''
    type: str
    uid: str
    url: str = ''
