import gws
import gws.types as t


class Error(gws.Error):
    pass


class Request(gws.Data):
    alwaysXY: bool
    bounds: gws.Bounds
    crs: gws.ICrs
    isSoap: bool = False
    project: gws.IProject
    req: gws.IWebRequester
    service: gws.IOwsService
    targetCrs: gws.ICrs
    version: str
    xmlElement: t.Optional[gws.IXmlElement] = None


class LayerCaps(gws.Data):
    layer: gws.ILayer

    hasLegend: bool
    hasSearch: bool

    layerName: str
    layerQname: str
    featureName: str
    featureQname: str
    geometryName: str
    geometryQname: str

    maxScale: int
    minScale: int
    bounds: list[gws.Bounds]

    children: list['LayerCaps']
    ancestors: list['LayerCaps']

    model: t.Optional[gws.IModel]


class LayerCapsTree(gws.Data):
    root: t.Optional[LayerCaps]
    roots: list[LayerCaps]
    leaves: list[LayerCaps]


class FeatureCollectionMember(gws.Data):
    feature: gws.IFeature
    options: gws.LayerOwsOptions


class FeatureCollection(gws.Data):
    members: list[FeatureCollectionMember]
    timestamp: str
    numMatched: int
    numReturned: int
