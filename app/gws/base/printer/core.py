import gws
import gws.base.feature
import gws.base.template
import gws.lib.job
import gws.lib.style
import gws.types as t


class Config(gws.Config):
    """Printer configuration"""

    templates: list[gws.ext.config.template]
    """print templates"""


class Props(gws.Data):
    templates: list[gws.base.template.Props]


class Object(gws.Node):
    templates: list[gws.ITemplate]

    def configure(self):
        self.templates = self.create_children(gws.ext.object.template, self.cfg('templates'))

    def props(self, user):
        return gws.Props(
            templates=[tpl for tpl in self.templates if user.can_use(tpl)]
        )


##

class State(t.Enum):
    init = 'init'
    open = 'open'
    running = 'running'
    complete = 'complete'
    error = 'error'
    cancel = 'cancel'


class StatusRequest(gws.Request):
    jobUid: str


class StatusResponse(gws.Response):
    jobUid: str
    progress: int
    state: State
    stepType: str
    stepName: str
    url: str


##

class PlaneType(gws.Enum):
    bitmap = 'bitmap'
    url = 'url'
    features = 'features'
    raster = 'raster'
    vector = 'vector'
    soup = 'soup'


class Plane(gws.Data):
    type: PlaneType

    opacity: t.Optional[float]
    cssSelector: t.Optional[str]

    bitmapData: t.Optional[bytes]
    bitmapMode: t.Optional[str]
    bitmapWidth: t.Optional[int]
    bitmapHeight: t.Optional[int]

    url: t.Optional[str]

    features: t.Optional[list[gws.FeatureProps]]

    layerUid: t.Optional[str]
    subLayers: t.Optional[list[str]]

    soupPoints: t.Optional[list[gws.Point]]
    soupTags: t.Optional[list[t.Any]]
    soupStyles: t.Optional[list[gws.lib.style.Props]]


class MapParams(gws.Data):
    backgroundColor: t.Optional[int]
    bbox: t.Optional[gws.Extent]
    center: t.Optional[gws.Point]
    planes: list[Plane]
    rotation: t.Optional[int]
    scale: int
    styles: t.Optional[list[gws.lib.style.Props]]
    visibleLayers: t.Optional[list[str]]


class RequestType(gws.Enum):
    template = 'template'
    map = 'map'


class Request(gws.Request):
    type: RequestType

    args: t.Optional[dict]
    crs: t.Optional[gws.CrsName]
    outputFormat: t.Optional[str]
    maps: t.Optional[list[MapParams]]

    templateUid: t.Optional[str]
    dpi: t.Optional[int]
    outputSize: t.Optional[gws.Size]
