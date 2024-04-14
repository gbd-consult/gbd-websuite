import gws
import gws.types as t


class Error(gws.Error):
    pass


class Request(gws.Data):
    alwaysXY: bool
    bounds: gws.Bounds
    crs: gws.Crs
    isSoap: bool = False
    project: gws.Project
    req: gws.WebRequester
    service: gws.OwsService
    targetCrs: gws.Crs
    version: str
    xmlElement: t.Optional[gws.XmlElement] = None


class LayerCaps(gws.Data):
    layer: gws.Layer

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

    model: t.Optional[gws.Model]


class LayerCapsTree(gws.Data):
    root: t.Optional[LayerCaps]
    roots: list[LayerCaps]
    leaves: list[LayerCaps]


class FeatureCollectionMember(gws.Data):
    feature: gws.Feature
    options: gws.LayerOwsOptions


class FeatureCollection(gws.Data):
    members: list[FeatureCollectionMember]
    timestamp: str
    numMatched: int
    numReturned: int
