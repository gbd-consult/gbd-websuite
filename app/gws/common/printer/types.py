import gws.types as t
import gws.tools.job


class PrintFeatureProps(t.Data):
    label: str = ''
    shape: t.Optional[t.ShapeProps]
    style: t.Optional[t.StyleProps]


class PrintItem(t.Data):
    bitmap: t.Optional[str]
    features: t.Optional[t.List[PrintFeatureProps]]
    subLayers: t.Optional[t.List[str]]
    layerUid: t.Optional[str]
    opacity: t.Optional[float]
    printAsVector: t.Optional[bool]
    style: t.Optional[t.StyleProps]


class PrintSection(t.Data):
    center: t.Point
    data: t.Optional[dict]
    items: t.Optional[t.List[PrintItem]]


class PrintParams(t.Data):
    projectUid: str
    items: t.List[PrintItem]
    rotation: int
    scale: int
    format: t.Optional[str]
    templateUid: str
    sections: t.Optional[t.List[PrintSection]]
    quality: int
    mapWidth: t.Optional[int]
    mapHeight: t.Optional[int]


class PrinterQueryParams(t.Data):
    jobUid: str


class PrinterResponse(t.Response):
    jobUid: str = ''
    progress: int = 0
    state: gws.tools.job.State
    otype: str = ''
    oname: str = ''
    url: str = ''
