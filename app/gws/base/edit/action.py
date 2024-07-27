"""Backend for vector edit operations."""

from typing import Optional

import gws
import gws.base.action
import gws.lib.mime
from . import core

gws.ext.new.action('edit')


class Config(gws.base.action.Config):
    """Edit action"""
    pass


class Props(gws.base.action.Props):
    pass


class Object(core.Action):
    @gws.ext.command.api('editGetModels')
    def edit_get_models(self, req: gws.WebRequester, p: core.GetModelsRequest) -> core.GetModelsResponse:
        return self.get_models_response(req, p)

    @gws.ext.command.api('editGetFeatures')
    def edit_get_features(self, req: gws.WebRequester, p: core.GetFeaturesRequest) -> core.FeatureListResponse:
        return self.get_features_response(req, p)

    @gws.ext.command.api('editGetRelatableFeatures')
    def edit_get_relatable_features(self, req: gws.WebRequester, p: core.GetRelatableFeaturesRequest) -> core.FeatureListResponse:
        return self.get_relatable_features_response(req, p)

    @gws.ext.command.api('editGetFeature')
    def edit_get_feature(self, req: gws.WebRequester, p: core.GetFeatureRequest) -> core.FeatureResponse:
        return self.get_feature_response(req, p)

    @gws.ext.command.api('editInitFeature')
    def edit_init_feature(self, req: gws.WebRequester, p: core.InitFeatureRequest) -> core.FeatureResponse:
        return self.init_feature_response(req, p)

    @gws.ext.command.api('editWriteFeature')
    def edit_write_feature(self, req: gws.WebRequester, p: core.WriteFeatureRequest) -> core.WriteResponse:
        return self.write_feature_response(req, p)

    @gws.ext.command.api('editDeleteFeature')
    def edit_delete_feature(self, req: gws.WebRequester, p: core.DeleteFeatureRequest) -> gws.Response:
        return self.delete_feature_response(req, p)
