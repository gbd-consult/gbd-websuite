from typing import Optional

import gws
import gws.lib.mime


class GetModelsRequest(gws.Request):
    pass


class GetModelsResponse(gws.Response):
    models: list[gws.ext.props.model]


class GetFeaturesRequest(gws.Request):
    modelUids: list[str]
    crs: Optional[gws.CrsName]
    extent: Optional[gws.Extent]
    featureUids: Optional[list[str]]
    keyword: Optional[str]
    resolution: Optional[float]
    shapes: Optional[list[gws.ShapeProps]]
    tolerance: Optional[str]


class GetFeaturesResponse(gws.Response):
    features: list[gws.FeatureProps]


class GetRelatableFeaturesRequest(gws.Request):
    modelUid: str
    fieldName: str
    extent: Optional[gws.Extent]
    keyword: Optional[str]


class GetRelatableFeaturesResponse(gws.Response):
    features: list[gws.FeatureProps]


class GetFeatureRequest(gws.Request):
    modelUid: str
    featureUid: str


class GetFeatureResponse(gws.Response):
    feature: gws.FeatureProps


class InitFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class InitFeatureResponse(gws.Response):
    feature: gws.FeatureProps


class WriteFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class WriteFeatureResponse(gws.Response):
    validationErrors: list[gws.ModelValidationError]
    feature: gws.FeatureProps


class DeleteFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class DeleteFeatureResponse(gws.Response):
    pass
