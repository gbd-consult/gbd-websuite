import gws
import gws.base.action
import gws.base.layer
import gws.base.legend
import gws.gis.cache
import gws.gis.crs
import gws.base.feature
import gws.lib.image
import gws.lib.jsonx
import gws.lib.mime
import gws.gis.render
import gws.lib.uom as units
import gws.types as t

gws.ext.new.action('edit')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class GetBoxRequest(gws.Request):
    bbox: gws.Extent
    width: int
    height: int
    layerUid: str
    crs: t.Optional[gws.CrsName]
    dpi: t.Optional[int]
    layers: t.Optional[t.List[str]]


class GetXyzRequest(gws.Request):
    layerUid: str
    x: int
    y: int
    z: int


class GetLegendRequest(gws.Request):
    layerUid: str


class ImageResponse(gws.Response):
    content: bytes
    mime: str


class DescribeLayerRequest(gws.Request):
    layerUid: str


class DescribeLayerResponse(gws.Request):
    description: str


class GetFeaturesRequest(gws.Request):
    bbox: t.Optional[gws.Extent]
    layerUid: str
    crs: t.Optional[gws.CrsName]
    resolution: t.Optional[float]
    limit: int = 0


class GetFeaturesResponse(gws.Response):
    features: t.List[gws.FeatureProps]


class Object(gws.base.action.Object):
    @gws.ext.command.api('editGetLayers')
    def api_get_layers(self, req: gws.IWebRequester, p: GetXyzRequest) -> ImageResponse:
        """Get an XYZ tile"""

        ls = []

        la: gws.ILayer
        for la in self.root.find_all(gws.ext.object.layer):
            if not la.layers and req.user.can_use(la):
                model = la.modelMgr.model_for(req.user, allow=gws.WRITE)
                if model:
                    ls.append(dict(
                        layerUid=la.uid,
                        modelUid=model.uid
                    ))
