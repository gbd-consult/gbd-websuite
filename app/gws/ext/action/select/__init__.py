import time

import gws
import gws.tools.storage

import gws.types as t


class SaveFeaturesParams(t.Data):
    name: str
    features: t.List[t.FeatureProps]


class SaveFeaturesResponse(t.Response):
    names: t.List[str]


class LoadFeaturesParams(t.Data):
    name: str


class LoadFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class GetSaveNamesParams(t.Data):
    pass


class GetSaveNamesResponse(t.Response):
    names: t.List[str]


class Config(t.WithTypeAndAccess):
    """select action"""
    pass


class Object(gws.Object):

    def api_save_features(self, req, p: SaveFeaturesParams) -> SaveFeaturesResponse:
        gws.tools.storage.put('features', p.name, req.user.full_uid, p.features)
        names = gws.tools.storage.get_names('features', req.user.full_uid)
        return SaveFeaturesResponse({
            'names': names
        })

    def api_load_features(self, req, p: LoadFeaturesParams) -> LoadFeaturesResponse:
        fs = gws.tools.storage.get('features', p.name, req.user.full_uid)
        return LoadFeaturesResponse({
            'features': fs or []
        })

    def api_get_save_names(self, req, p: GetSaveNamesParams) -> GetSaveNamesResponse:
        names = gws.tools.storage.get_names('features', req.user.full_uid)
        return GetSaveNamesResponse({
            'names': names
        })
