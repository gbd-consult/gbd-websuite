"""Base data structures for OWS services."""

from typing import Optional

import gws


class Error(gws.Error):
    """OWS server error."""

    pass


class LayerCaps(gws.Data):
    """Layer wrapper object.

    A ``LayerCaps`` object wraps a ``Layer`` object and provides
    additional data needed to represent a layer in an OWS service.
    """

    layer: gws.Layer
    """Wrapped layer."""
    title: str
    """Layer title."""

    isGroup: bool
    """The layer is a group."""
    hasLegend: bool
    """The layer has a legend."""
    isSearchable: bool
    """The layer can be searched."""

    layerName: str
    """Layer name in the service."""
    featureName: str
    """Feature type name in the service, without a namespace prefix."""
    geometryName: str
    """Name of the geometry attribute."""

    maxScale: int
    """Max. scale denominator of the layer."""
    minScale: int
    """Min. scale denominator of the layer."""
    bounds: list[gws.Bounds]
    """Layer extent in each CRS supported by the service."""

    children: list['LayerCaps']
    """Direct children of a group."""
    leaves: list['LayerCaps']
    """All non-group descendants of a group."""

    model: Optional[gws.Model]
    """Model to read the layer features, if any."""
    xmlNamespace: Optional[gws.XmlNamespace]
    """XML namespace of the feature type, if any."""


class FeatureCollectionMember(gws.Data):
    """A member of a feature collection."""

    feature: gws.Feature
    """Feature."""
    layer: Optional[gws.Layer]
    """Layer the feature was found in."""
    layerCaps: Optional[LayerCaps]
    """Caps of that layer, if it is part of the request."""


class FeatureCollection(gws.Data):
    """Feature collection, the result of a search request."""

    members: list[FeatureCollectionMember]
    """Collection members."""
    values: list
    """Property values, for value requests like ``GetPropertyValue``."""
    timestamp: str
    """Time of the response, as an ISO string."""
    numMatched: int
    """Total number of matching features."""
    numReturned: int
    """Number of returned features."""


class MetadataCollection(gws.Data):
    """Metadata collection, the result of a catalog request."""

    members: list[gws.Metadata]
    """Metadata records."""
    timestamp: str
    """Time of the response, as an ISO string."""
    numMatched: int
    """Total number of matching records."""
    numReturned: int
    """Number of returned records."""
    nextRecord: int
    """Position of the next record, for paging."""


IMAGE_VERBS = {
    gws.OwsVerb.GetMap,
    gws.OwsVerb.GetTile,
    gws.OwsVerb.GetLegendGraphic,
}
"""OWS verbs which are supposed to return images."""
