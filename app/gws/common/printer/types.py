import gws.types as t
import gws.tools.job


class PrintItemBase(t.Data):
    opacity: float = 1
    style: t.Optional[t.StyleProps]
    printAsVector: t.Optional[bool]


class PrintItemBitmap(PrintItemBase):
    type: str = 'bitmap'
    data: bytes
    mode: str
    width: int
    height: int


class PrintItemUrl(PrintItemBase):
    type: str = 'url'
    url: str


class PrintItemFeatures(PrintItemBase):
    type: str = 'features'
    features: t.List[t.FeatureProps]


class PrintItemLayer(PrintItemBase):
    type: str = 'layer'
    layerUid: str
    subLayers: t.Optional[t.List[str]]


class PrintItemFragment(PrintItemBase):
    type: str = 'fragment'
    fragment: t.SvgFragment


#:alias
PrintItem = t.Union[
    PrintItemBitmap,
    PrintItemUrl,
    PrintItemFeatures,
    PrintItemLayer,
    PrintItemFragment
]


class PrintSection(t.Data):
    center: t.Point
    attributes: t.Optional[t.List[t.Attribute]]
    items: t.Optional[t.List[PrintItem]]


class PrintParamsBase(t.Params):
    crs: t.Optional[t.Crs]
    format: t.Optional[str]
    items: t.List[PrintItem]
    rotation: int = 0
    scale: int
    sections: t.Optional[t.List[PrintSection]]


class PrintParamsWithTemplate(PrintParamsBase):
    type = 'template'
    quality: int
    templateUid: str


class PrintParamsWithMap(PrintParamsBase):
    type = 'map'
    dpi: int
    mapHeight: int
    mapWidth: int


#:alias
PrintParams = t.Union[PrintParamsWithTemplate, PrintParamsWithMap]


class PrinterQueryParams(t.Params):
    jobUid: str


class PrinterResponse(t.Response):
    jobUid: str = ''
    progress: int = 0
    state: gws.tools.job.State
    steptype: str = ''
    stepname: str = ''
    url: str = ''
