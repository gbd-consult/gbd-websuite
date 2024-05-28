from typing import Optional, Callable

import gws


class Error(gws.Error):
    pass


class ServiceRequest(gws.Data):
    alwaysXY: bool
    bounds: gws.Bounds
    crs: gws.Crs
    isSoap: bool = False
    project: Optional[gws.Project]
    req: gws.WebRequester
    service: gws.OwsService
    targetCrs: gws.Crs
    verb: gws.OwsVerb
    version: str
    xmlElement: Optional[gws.XmlElement]


class LayerCaps(gws.Data):
    layer: gws.Layer
    title: str

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

    model: Optional[gws.Model]
    xmlNamespace: Optional[gws.XmlNamespace]


class LayerCapsTree(gws.Data):
    root: Optional[LayerCaps]
    roots: list[LayerCaps]
    leaves: list[LayerCaps]


class FeatureCollectionMember(gws.Data):
    feature: gws.Feature
    layer: Optional[gws.Layer]
    layerCaps: Optional[LayerCaps]


class FeatureCollection(gws.Data):
    members: list[FeatureCollectionMember]
    timestamp: str
    numMatched: int
    numReturned: int


class TemplateArgs(gws.TemplateArgs):
    """Arguments for service templates."""

    featureCollection: FeatureCollection
    layerCapsTree: LayerCapsTree
    layerCapsList: list[LayerCaps]
    sr: ServiceRequest
    service: gws.OwsService
    serviceUrl: str
    url_for: Callable
    gmlVersion: int
    version: str
