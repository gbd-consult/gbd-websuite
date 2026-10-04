"""Edit action."""

from typing import Optional, cast

import gws
import gws.base.action

from . import api, helper


@gws.ext.config.action('edit')
class Config(gws.base.action.Config):
    """Feature editing in the client."""
    pass


@gws.ext.props.action('edit')
class Props(gws.base.action.Props):
    pass


@gws.ext.object.action('edit')
class Object(gws.base.action.Object):
    """Edit action.

    Delegates all work to the ``edit`` helper.
    """

    h: helper.Object
    """The ``edit`` helper."""

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('edit'))

    @gws.ext.command.api('editGetModels')
    def api_get_models(self, req: gws.WebRequester, p: api.GetModelsRequest) -> api.GetModelsResponse:
        """Return the editable models of a project."""
        return self.h.get_models_response(req, p, self.h.get_models(req, p))

    @gws.ext.command.api('editGetFeatures')
    def api_get_features(self, req: gws.WebRequester, p: api.GetFeaturesRequest) -> api.GetFeaturesResponse:
        """Return features of editable models."""
        return self.h.get_features_response(req, p, self.h.get_features(req, p))

    @gws.ext.command.api('editGetRelatableFeatures')
    def api_get_relatable_features(self, req: gws.WebRequester, p: api.GetRelatableFeaturesRequest) -> api.GetRelatableFeaturesResponse:
        """Return features that can be linked in a related field."""
        return self.h.get_relatable_features_response(req, p, self.h.get_relatable_features(req, p))

    @gws.ext.command.api('editGetFeature')
    def api_get_feature(self, req: gws.WebRequester, p: api.GetFeatureRequest) -> api.GetFeatureResponse:
        """Return a single feature for the edit form."""
        return self.h.get_feature_response(req, p, self.h.get_feature(req, p))

    @gws.ext.command.api('editInitFeature')
    def api_init_feature(self, req: gws.WebRequester, p: api.InitFeatureRequest) -> api.InitFeatureResponse:
        """Return a new feature with initial values, without saving it."""
        return self.h.init_feature_response(req, p, self.h.init_feature(req, p))

    @gws.ext.command.api('editWriteFeature')
    def api_write_feature(self, req: gws.WebRequester, p: api.WriteFeatureRequest) -> api.WriteFeatureResponse:
        """Validate and save a new or existing feature."""
        return self.h.write_feature_response(req, p, self.h.write_feature(req, p))

    @gws.ext.command.api('editDeleteFeature')
    def api_delete_feature(self, req: gws.WebRequester, p: api.DeleteFeatureRequest) -> api.DeleteFeatureResponse:
        """Delete a feature."""
        return self.h.delete_feature_response(req, p, self.h.delete_feature(req, p))
