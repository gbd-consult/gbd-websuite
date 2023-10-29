import gws
import gws.base.feature
import gws.base.model
import gws.base.template
import gws.config.util
import gws.lib.job
import gws.lib.style
import gws.types as t

gws.ext.new.printer('default')


class Config(gws.Config):
    """Printer configuration"""

    template: gws.ext.config.template
    """Print template"""
    title: str = ''
    """Printer title"""
    models: t.Optional[list[gws.ext.config.model]]
    """Data models"""
    qualityLevels: t.Optional[list[gws.TemplateQualityLevel]]
    """Quality levels supported by this printer"""


class Props(gws.Props):
    template: gws.base.template.Props
    model: gws.base.model.Props
    qualityLevels: list[gws.TemplateQualityLevel]
    title: str


class Object(gws.Node, gws.IPrinter):

    def configure(self):
        gws.config.util.configure_models(self)
        self.template = self.create_child(gws.ext.object.template, self.cfg('template'))
        self.qualityLevels = self.cfg('qualityLevels') or [gws.TemplateQualityLevel(name='default', dpi=0)]
        self.title = self.cfg('title') or self.template.title or ''

    def props(self, user):
        model = self.root.app.modelMgr.locate_model(self, user=user, access=gws.Access.write)
        return Props(
            uid=self.uid,
            template=self.template,
            model=model,
            qualityLevels=self.qualityLevels,
            title=self.title,
        )


##

class StatusResponse(gws.Response):
    jobUid: str
    progress: int
    state: gws.JobState
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

    printerUid: t.Optional[str]
    dpi: t.Optional[int]
    outputSize: t.Optional[gws.Size]
