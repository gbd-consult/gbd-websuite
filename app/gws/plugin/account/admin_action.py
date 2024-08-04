"""Account admin action."""

from typing import Optional, cast

import gws
import gws.config.util
import gws.base.action
import gws.lib.mime
import gws.base.edit.api as api

from . import helper

gws.ext.new.action('accountadmin')


class Config(gws.base.action.Config):
    """Account Admin action"""

    models: Optional[list[gws.ext.config.model]]
    """Account data models."""


class Props(gws.base.action.Props):
    pass


##


class ResetRequest(gws.Request):
    featureUid: str


class ResetResponse(gws.Response):
    feature: gws.FeatureProps


class Object(gws.base.action.Object):
    h: helper.Object

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('account'))

    @gws.ext.command.api('accountadminGetModels')
    def api_get_models(self, req: gws.WebRequester, p: api.GetModelsRequest) -> api.GetModelsResponse:
        return self.h.get_models_response(req, p, self.h.get_models(req, p))

    @gws.ext.command.api('accountadminGetFeatures')
    def api_get_features(self, req: gws.WebRequester, p: api.GetFeaturesRequest) -> api.GetFeaturesResponse:
        return self.h.get_features_response(req, p, self.h.get_features(req, p))

    @gws.ext.command.api('accountadminGetRelatableFeatures')
    def api_get_relatable_features(self, req: gws.WebRequester, p: api.GetRelatableFeaturesRequest) -> api.GetRelatableFeaturesResponse:
        return self.h.get_relatable_features_response(req, p, self.h.get_relatable_features(req, p))

    @gws.ext.command.api('accountadminGetFeature')
    def api_get_feature(self, req: gws.WebRequester, p: api.GetFeatureRequest) -> api.GetFeatureResponse:
        return self.h.get_feature_response(req, p, self.h.get_feature(req, p))

    @gws.ext.command.api('accountadminInitFeature')
    def api_init_feature(self, req: gws.WebRequester, p: api.InitFeatureRequest) -> api.InitFeatureResponse:
        return self.h.init_feature_response(req, p, self.h.init_feature(req, p))

    @gws.ext.command.api('accountadminWriteFeature')
    def api_write_feature(self, req: gws.WebRequester, p: api.WriteFeatureRequest) -> api.WriteFeatureResponse:
        return self.h.write_feature_response(req, p, self.h.write_feature(req, p))

    @gws.ext.command.api('accountadminDeleteFeature')
    def api_delete_feature(self, req: gws.WebRequester, p: api.DeleteFeatureRequest) -> api.DeleteFeatureResponse:
        return self.h.delete_feature_response(req, p, self.h.delete_feature(req, p))

    @gws.ext.command.api('accountadminReset')
    def api_reset(self, req: gws.WebRequester, p: ResetRequest) -> ResetResponse:
        uid = p.featureUid
        account = self.h.get_account_by_id(uid)
        if not account:
            raise gws.NotFoundError()
        self.h.reset(account)

        mc = self.h.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        fs = self.h.adminModel.get_features([self.h.get_uid(account)], mc)
        return ResetResponse(feature=self.h.feature_to_props(fs[0], mc))
