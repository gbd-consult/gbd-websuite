import gws
import gws.types as t
import gws.lib.feature
import gws.lib.style
import gws.lib.job



class StatusParams(gws.Params):
    jobUid: str


class StatusResponse(gws.Response):
    jobUid: str
    progress: int
    state: gws.lib.job.State
    steptype: str
    stepname: str
    url: str


class ItemBase(gws.Data):
    opacity: t.Optional[float]
    style: t.Optional[gws.lib.style.Props]


class ItemBitmap(ItemBase):
    type: t.Literal['bitmap']
    data: bytes
    mode: str
    width: int
    height: int


class ItemUrl(ItemBase):
    type: t.Literal['url']
    url: str


class ItemFeatures(ItemBase):
    type: t.Literal['features']
    features: t.List[gws.lib.feature.Props]


class ItemRaster(ItemBase):
    type: t.Literal['raster']
    layerUid: str
    subLayers: t.Optional[t.List[str]]


class ItemVector(ItemBase):
    type: t.Literal['vector']
    layerUid: str


class ItemFragment(ItemBase):
    type: t.Literal['fragment']
    points: t.List[gws.Point]
    styles: t.Optional[t.List[gws.lib.style.Props]]
    tags: t.List[t.Any]


#: Print item (Variant)
Item = t.Union[
    ItemBitmap,
    ItemUrl,
    ItemFeatures,
    ItemRaster,
    ItemVector,
    ItemFragment
]


class Section(gws.Data):
    center: gws.Point
    context: t.Optional[dict]
    items: t.Optional[t.List[Item]]


class ParamsBase(gws.Params):
    crs: t.Optional[gws.Crs]
    format: t.Optional[str]
    items: t.List[Item]
    legendLayers: t.Optional[t.List[str]]
    rotation: int = 0
    scale: int
    sections: t.Optional[t.List[Section]]


class ParamsWithTemplate(ParamsBase):
    type: t.Literal['template']
    quality: int
    templateUid: str


class ParamsWithMap(ParamsBase):
    type: t.Literal['map']
    dpi: int
    mapHeight: int
    mapWidth: int


#: Print params (Variant)
Params = t.Union[ParamsWithTemplate, ParamsWithMap]
