"""Base data structures for OWS services."""

from typing import Optional

import gws


class Error(gws.Error):
    pass


class LayerCaps(gws.Data):
    """Layer wrapper object.

    A `LayerCaps` object wraps a `Layer` object and provides
    additional data needed to represent a layer in an OWS service.
    """

    layer: gws.Layer
    title: str

    isGroup: bool
    hasLegend: bool
    isSearchable: bool

    layerName: str
    layerNameQ: str
    featureName: str
    featureNameQ: str
    geometryName: str
    geometryNameQ: str

    maxScale: int
    minScale: int
    bounds: list[gws.Bounds]

    children: list['LayerCaps']
    leaves: list['LayerCaps']

    model: Optional[gws.Model]
    xmlNamespace: Optional[gws.XmlNamespace]


class FeatureCollectionMember(gws.Data):
    """A member of a Feature Collection."""

    feature: gws.Feature
    layer: Optional[gws.Layer]
    layerCaps: Optional[LayerCaps]


class FeatureCollection(gws.Data):
    """Feature Collection."""

    members: list[FeatureCollectionMember]
    values: list
    timestamp: str
    numMatched: int
    numReturned: int


IMAGE_VERBS = {
    gws.OwsVerb.GetMap,
    gws.OwsVerb.GetTile,
    gws.OwsVerb.GetLegendGraphic,
}
"""OWS verbs which are supposed to return images."""
