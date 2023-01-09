import gws
import gws.base.feature
import gws.base.template
import gws.lib.job
import gws.lib.style
import gws.types as t


class StatusParams(gws.Request):
    jobUid: str


class StatusResponse(gws.Response):
    jobUid: str
    progress: int
    state: gws.lib.job.State
    steptype: str
    stepname: str
    url: str


class PlaneBase(gws.Data):
    opacity: t.Optional[float]
    style: t.Optional[gws.lib.style.Props]


class PlaneBitmap(PlaneBase):
    type: t.Literal['bitmap']
    data: bytes
    mode: str
    width: int
    height: int


class PlaneUrl(PlaneBase):
    type: t.Literal['url']
    url: str


class PlaneFeatures(PlaneBase):
    type: t.Literal['features']
    features: t.List[gws.base.feature.Props]


class PlaneRaster(PlaneBase):
    type: t.Literal['raster']
    layerUid: str
    subLayers: t.Optional[t.List[str]]


class PlaneVector(PlaneBase):
    type: t.Literal['vector']
    layerUid: str


class PlaneSoup(PlaneBase):
    type: t.Literal['soup']
    points: t.List[gws.Point]
    tags: t.List[t.Any]
    styles: t.Optional[t.List[gws.lib.style.Props]]


Plane = t.Union[
    PlaneBitmap,
    PlaneUrl,
    PlaneFeatures,
    PlaneRaster,
    PlaneVector,
    PlaneSoup,
]
"""variant: Print plane"""


class MapParams(gws.Data):
    background_color: t.Optional[int]
    bbox: t.Optional[gws.Extent]
    center: t.Optional[gws.Point]
    planes: t.List[Plane]
    rotation: t.Optional[int]
    scale: int
    visibleLayers: t.Optional[t.List[str]]


class ParamsBase(gws.Request):
    context: t.Optional[dict]
    crs: t.Optional[gws.CrsName]
    outputFormat: t.Optional[str]
    maps: t.Optional[t.List[MapParams]]


class ParamsWithTemplate(ParamsBase):
    type: t.Literal['template']
    qualityLevel: int
    templateUid: str


class ParamsWithMap(ParamsBase):
    type: t.Literal['map']
    dpi: int
    outputSize: gws.Size


Params = t.Union[ParamsWithTemplate, ParamsWithMap]
"""variant: Print params"""


class Config(gws.Config):
    """Printer configuration"""

    templates: t.List[gws.ext.config.template]
    """print templates"""


class Props(gws.Data):
    templates: t.List[gws.base.template.Props]


class Object(gws.Node):
    templates: t.List[gws.ITemplate]

    def configure(self):
        self.create_children(gws.ext.object.template, self.var('templates'))

    def props(self, user):
        return gws.Props(
            templates=[tpl for tpl in self.templates if user.can_use(tpl)]
        )
