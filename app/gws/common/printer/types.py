import gws.types as t
import gws.tools.job


class PrintItemBase(t.Data):
    opacity: t.Optional[float]
    style: t.Optional[t.StyleProps]


class PrintItemBitmap(PrintItemBase):
    type: t.Literal = 'bitmap'
    data: bytes
    mode: str
    width: int
    height: int


class PrintItemUrl(PrintItemBase):
    type: t.Literal = 'url'
    url: str


class PrintItemFeatures(PrintItemBase):
    type: t.Literal = 'features'
    features: t.List[t.FeatureProps]


class PrintItemRaster(PrintItemBase):
    type: t.Literal = 'raster'
    layerUid: str
    subLayers: t.Optional[t.List[str]]


class PrintItemVector(PrintItemBase):
    type: t.Literal = 'vector'
    layerUid: str


class PrintItemFragment(PrintItemBase):
    type: t.Literal = 'fragment'
    points: t.List[t.Point]
    styles: t.Optional[t.List[t.StyleProps]]
    tags: t.List[t.Any]


#:alias
PrintItem = t.Union[
    PrintItemBitmap,
    PrintItemUrl,
    PrintItemFeatures,
    PrintItemRaster,
    PrintItemVector,
    PrintItemFragment
]


class PrintSection(t.Data):
    center: t.Point
    context: t.Optional[dict]
    items: t.Optional[t.List[PrintItem]]


class PrintParamsBase(t.Params):
    crs: t.Optional[t.Crs]
    format: t.Optional[str]
    items: t.List[PrintItem]
    legendLayers: t.Optional[t.List[str]]
    rotation: int = 0
    scale: int
    sections: t.Optional[t.List[PrintSection]]


class PrintParamsWithTemplate(PrintParamsBase):
    type: t.Literal = 'template'
    quality: int
    templateUid: str


class PrintParamsWithMap(PrintParamsBase):
    type: t.Literal = 'map'
    dpi: int
    mapHeight: int
    mapWidth: int


#:alias
PrintParams = t.Union[PrintParamsWithTemplate, PrintParamsWithMap]
