"""Request and response types of the edit API."""

from typing import Optional

import gws
import gws.lib.mime


class GetModelsRequest(gws.Request):
    """Request for the editable models of a project."""

    pass


class GetModelsResponse(gws.Response):
    """Editable models of a project."""

    models: list[gws.ext.props.model]
    """Model props."""


class GetFeaturesRequest(gws.Request):
    """Request for features of editable models."""

    modelUids: list[str]
    """Uids of the models to search."""
    crs: Optional[gws.CrsName]
    """CRS of ``extent``, the project map CRS by default."""
    extent: Optional[gws.Extent]
    """Search extent."""
    featureUids: Optional[list[str]]
    """Return only features with these uids."""
    keyword: Optional[str]
    """Search keyword."""
    resolution: Optional[float]
    """Map resolution."""
    shapes: Optional[list[gws.ShapeProps]]
    """Search shapes, combined into one."""
    tolerance: Optional[str]
    """Search tolerance. Not used, the helper applies a fixed tolerance of 10 pixels."""


class GetFeaturesResponse(gws.Response):
    """Features of editable models."""

    features: list[gws.FeatureProps]
    """Feature props."""


class GetRelatableFeaturesRequest(gws.Request):
    """Request for features that can be linked in a related field."""

    modelUid: str
    """Uid of the model that has the field."""
    fieldName: str
    """Name of the related field."""
    extent: Optional[gws.Extent]
    """Search extent. Not used."""
    keyword: Optional[str]
    """Search keyword."""


class GetRelatableFeaturesResponse(gws.Response):
    """Features that can be linked in a related field."""

    features: list[gws.FeatureProps]
    """Feature props."""


class GetFeatureRequest(gws.Request):
    """Request for a single feature."""

    modelUid: str
    """Model uid."""
    featureUid: str
    """Feature uid."""


class GetFeatureResponse(gws.Response):
    """A single feature."""

    feature: gws.FeatureProps
    """Feature props."""


class InitFeatureRequest(gws.Request):
    """Request to initialize a new feature."""

    modelUid: str
    """Model uid."""
    feature: gws.FeatureProps
    """Props of the new feature."""


class InitFeatureResponse(gws.Response):
    """A new feature with initial values."""

    feature: gws.FeatureProps
    """Feature props."""


class WriteFeatureRequest(gws.Request):
    """Request to save a new or existing feature."""

    modelUid: str
    """Model uid."""
    feature: gws.FeatureProps
    """Feature props. ``isNew`` decides between create and update."""


class WriteFeatureResponse(gws.Response):
    """Result of saving a feature."""

    validationErrors: list[gws.ModelValidationError]
    """Validation errors. If not empty, the feature was not saved."""
    feature: gws.FeatureProps
    """The saved feature, read back from the model."""


class DeleteFeatureRequest(gws.Request):
    """Request to delete a feature."""

    modelUid: str
    """Model uid."""
    feature: gws.FeatureProps
    """Props of the feature to delete."""


class DeleteFeatureResponse(gws.Response):
    """Result of deleting a feature."""

    pass
