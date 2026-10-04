"""Action for account administration."""

from typing import Optional, cast

import gws
import gws.config.util
import gws.base.action
import gws.lib.mime
import gws.base.edit.api as api

from . import helper


@gws.ext.config.action('accountadmin')
class Config(gws.base.action.Config):
    """Administration of user accounts by account administrators."""

    models: Optional[list[gws.ext.config.model]]
    """Account data models."""


@gws.ext.props.action('accountadmin')
class Props(gws.base.action.Props):
    pass


##


class ResetRequest(gws.Request):
    """Request to reset an account."""

    featureUid: str
    """Uid of the account feature."""


class ResetResponse(gws.Response):
    """Response to an account reset."""

    feature: gws.FeatureProps
    """The account feature after the reset."""


@gws.ext.object.action('accountadmin')
class Object(gws.base.action.Object):
    """Account administration action.

    Provides the edit API for the helper's ``adminModel`` and the account reset.
    """

    h: helper.Object
    """The account helper."""

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('account'))

    @gws.ext.command.api('accountadminGetModels')
    def api_get_models(self, req: gws.WebRequester, p: api.GetModelsRequest) -> api.GetModelsResponse:
        """Return the account administration model."""

        return self.h.get_models_response(req, p, self.h.get_models(req, p))

    @gws.ext.command.api('accountadminGetFeatures')
    def api_get_features(self, req: gws.WebRequester, p: api.GetFeaturesRequest) -> api.GetFeaturesResponse:
        """Return account features."""

        return self.h.get_features_response(req, p, self.h.get_features(req, p))

    @gws.ext.command.api('accountadminGetRelatableFeatures')
    def api_get_relatable_features(self, req: gws.WebRequester, p: api.GetRelatableFeaturesRequest) -> api.GetRelatableFeaturesResponse:
        """Return features that can be linked to an account."""

        return self.h.get_relatable_features_response(req, p, self.h.get_relatable_features(req, p))

    @gws.ext.command.api('accountadminGetFeature')
    def api_get_feature(self, req: gws.WebRequester, p: api.GetFeatureRequest) -> api.GetFeatureResponse:
        """Return an account feature."""

        return self.h.get_feature_response(req, p, self.h.get_feature(req, p))

    @gws.ext.command.api('accountadminInitFeature')
    def api_init_feature(self, req: gws.WebRequester, p: api.InitFeatureRequest) -> api.InitFeatureResponse:
        """Return a new account feature with initial values, without saving it."""

        return self.h.init_feature_response(req, p, self.h.init_feature(req, p))

    @gws.ext.command.api('accountadminWriteFeature')
    def api_write_feature(self, req: gws.WebRequester, p: api.WriteFeatureRequest) -> api.WriteFeatureResponse:
        """Save an account feature."""

        return self.h.write_feature_response(req, p, self.h.write_feature(req, p))

    @gws.ext.command.api('accountadminDeleteFeature')
    def api_delete_feature(self, req: gws.WebRequester, p: api.DeleteFeatureRequest) -> api.DeleteFeatureResponse:
        """Delete an account feature."""

        return self.h.delete_feature_response(req, p, self.h.delete_feature(req, p))

    @gws.ext.command.api('accountadminReset')
    def api_reset(self, req: gws.WebRequester, p: ResetRequest) -> ResetResponse:
        """Reset an account."""

        if not req.user.can_write(self.h.adminModel):
            raise gws.ForbiddenError(f'account reset forbidden for {req.user.uid!r}')

        uid = p.featureUid
        account = self.h.get_account_by_id(uid)
        if not account:
            raise gws.NotFoundError()
        self.h.reset(account)

        mc = self.h.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        fs = self.h.adminModel.get_features([self.h.get_uid(account)], mc)
        return ResetResponse(feature=self.h.feature_to_props(fs[0], mc))
