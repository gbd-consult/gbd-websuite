"""Backend for vector edit operations."""

from typing import Optional, cast

import gws
import gws.base.action

from . import api, helper

gws.ext.new.action('edit')


class Config(gws.base.action.Config):
    """Edit action"""
    pass


class Props(gws.base.action.Props):
    pass


class Object(gws.base.action.Object):
    h: helper.Object

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('edit'))

    @gws.ext.command.api('editGetModels')
    def api_get_models(self, req: gws.WebRequester, p: api.GetModelsRequest) -> api.GetModelsResponse:
        return self.h.get_models_response(req, p, self.h.get_models(req, p))

    @gws.ext.command.api('editGetFeatures')
    def api_get_features(self, req: gws.WebRequester, p: api.GetFeaturesRequest) -> api.GetFeaturesResponse:
        return self.h.get_features_response(req, p, self.h.get_features(req, p))

    @gws.ext.command.api('editGetRelatableFeatures')
    def api_get_relatable_features(self, req: gws.WebRequester, p: api.GetRelatableFeaturesRequest) -> api.GetRelatableFeaturesResponse:
        return self.h.get_relatable_features_response(req, p, self.h.get_relatable_features(req, p))

    @gws.ext.command.api('editGetFeature')
    def api_get_feature(self, req: gws.WebRequester, p: api.GetFeatureRequest) -> api.GetFeatureResponse:
        return self.h.get_feature_response(req, p, self.h.get_feature(req, p))

    @gws.ext.command.api('editInitFeature')
    def api_init_feature(self, req: gws.WebRequester, p: api.InitFeatureRequest) -> api.InitFeatureResponse:
        return self.h.init_feature_response(req, p, self.h.init_feature(req, p))

    @gws.ext.command.api('editWriteFeature')
    def api_write_feature(self, req: gws.WebRequester, p: api.WriteFeatureRequest) -> api.WriteFeatureResponse:
        return self.h.write_feature_response(req, p, self.h.write_feature(req, p))

    @gws.ext.command.api('editDeleteFeature')
    def api_delete_feature(self, req: gws.WebRequester, p: api.DeleteFeatureRequest) -> api.DeleteFeatureResponse:
        return self.h.delete_feature_response(req, p, self.h.delete_feature(req, p))
