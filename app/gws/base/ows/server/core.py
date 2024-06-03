from typing import Optional

import gws


class Error(gws.Error):
    pass


class LayerCaps(gws.Data):
    layer: gws.Layer
    title: str

    isGroup: bool
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
    leaves: list['LayerCaps']

    model: Optional[gws.Model]
    xmlNamespace: Optional[gws.XmlNamespace]


class FeatureCollectionMember(gws.Data):
    feature: gws.Feature
    layer: Optional[gws.Layer]
    layerCaps: Optional[LayerCaps]


class FeatureCollection(gws.Data):
    members: list[FeatureCollectionMember]
    values: list
    timestamp: str
    numMatched: int
    numReturned: int
