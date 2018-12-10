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
    templateUid: str
    items: t.List[PrintItem]
    sections: t.List[PrintSection]
    rotation: int
    quality: int
    scale: int
    format: t.Optional[str]


class PrinterQueryParams(t.Data):
    jobUid: str


class PrinterResponse(t.Response):
    jobUid: str = ''
    progress: int = 0
    state: gws.tools.job.State
    otype: str = ''
    oname: str = ''
    url: str = ''
