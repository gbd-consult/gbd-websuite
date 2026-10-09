"""Edit helper."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.feature
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.lib.shape
import gws.base.template
import gws.base.web
import gws.lib.crs
import gws.gis.render
import gws.lib.image
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.uom

from . import api


LIST_VIEWS = ['title', 'label']
"""Feature views rendered for feature lists."""
DEFAULT_TOLERANCE = 10, gws.Uom.px
"""Search tolerance for feature searches."""


@gws.ext.object.helper('edit')
class Object(gws.Node):
    """Edit helper.

    Implements the edit API. For each command there is a method that does
    the work and returns models or features, and a method that converts the
    result into the API response. Other actions can call or override these
    methods separately.
    """

    def get_models(self, req: gws.WebRequester, p: api.GetModelsRequest) -> list[gws.Model]:
        """Return the editable models of the requested project.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            Models the user can edit, sorted by title.

        Raises:
            ``gws.NotFoundError``: If the project does not exist.
            ``gws.ForbiddenError``: If the user cannot read the project.
        """
        project = req.user.require_project(p.projectUid)
        return self.root.app.modelMgr.editable_models(project, req.user)

    def get_models_response(self, req: gws.WebRequester, p: gws.Request, models: list[gws.Model]) -> api.GetModelsResponse:
        """Create the response for ``editGetModels``.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            models: Models to return.

        Returns:
            The response with the model props.
        """
        return api.GetModelsResponse(
            models=gws.u.compact(gws.props_of(m, req.user) for m in models)
        )

    ##

    def get_features(self, req: gws.WebRequester, p: api.GetFeaturesRequest) -> list[gws.Feature]:
        """Search features in the requested models.

        The search is filtered by the extent, shapes, keyword and feature uids
        in the request.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            The found features, or an empty list if an extent is given without a CRS
            and the project has no map.

        Raises:
            ``gws.ForbiddenError``: If a model is not accessible or not editable.
        """
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList)

        search = gws.SearchQuery(project=mc.project, tolerance=DEFAULT_TOLERANCE)
        if p.extent:
            crs = p.crs
            if not crs and mc.project and mc.project.map:
                crs = mc.project.map.bounds.crs
            if not crs:
                gws.log.warning('no CRS specified for extent search')
                return []
            search.bounds = gws.Bounds(crs=crs, extent=p.extent)
        if p.shapes:
            shapes = [gws.lib.shape.from_props(s) for s in p.shapes]
            search.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])
        if p.tolerance:
            search.tolerance = gws.lib.uom.parse(p.tolerance, gws.Uom.px)
        if p.resolution:
            search.resolution = p.resolution
        if p.keyword:
            search.keyword = p.keyword
        if p.featureUids:
            search.uids = p.featureUids

        fs = []

        for model_uid in p.modelUids:
            model = self.require_model(model_uid, req.user, gws.Access.read)
            fs.extend(model.find_features(search, mc))

        return fs

    def get_features_response(self, req: gws.WebRequester, p: gws.Request, features: list[gws.Feature]) -> api.GetFeaturesResponse:
        """Create the response for ``editGetFeatures``.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            features: Features to return.

        Returns:
            The response with the feature props.
        """
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList)
        return api.GetFeaturesResponse(features=self.feature_list_to_props(features, mc))

    ##

    def get_relatable_features(self, req: gws.WebRequester, p: api.GetRelatableFeaturesRequest) -> list[gws.Feature]:
        """Return features that can be linked in a related field.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            Features found by the field for the request keyword.

        Raises:
            ``gws.ForbiddenError``: If the model or the field is not accessible.
        """
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList, max_depth=0)

        model = self.require_model(p.modelUid, req.user, gws.Access.read)
        field = self.require_field(model, p.fieldName, req.user, gws.Access.read)
        search = gws.SearchQuery(keyword=p.keyword)

        return field.find_relatable_features(search, mc)

    def get_relatable_features_response(self, req: gws.WebRequester, p: gws.Request, features: list[gws.Feature]) -> api.GetRelatableFeaturesResponse:
        """Create the response for ``editGetRelatableFeatures``.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            features: Features to return.

        Returns:
            The response with the feature props.
        """
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList)
        return api.GetRelatableFeaturesResponse(features=self.feature_list_to_props(features, mc))

    ##

    def get_feature(self, req: gws.WebRequester, p: api.GetFeatureRequest) -> Optional[gws.Feature]:
        """Return a single feature for the edit form.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            The feature, or ``None`` if it does not exist.

        Raises:
            ``gws.ForbiddenError``: If the model is not accessible or not editable.
        """
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        model = self.require_model(p.modelUid, req.user, gws.Access.read)
        fs = model.get_features([p.featureUid], mc)
        if fs:
            return fs[0]

    def get_feature_response(self, req: gws.WebRequester, p: gws.Request, feature: Optional[gws.Feature]) -> api.GetFeatureResponse:
        """Create the response for ``editGetFeature``.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            feature: Feature to return.

        Returns:
            The response with the feature props.

        Raises:
            ``gws.NotFoundError``: If ``feature`` is ``None``.
        """
        if not feature:
            raise gws.NotFoundError()
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        return api.GetFeatureResponse(feature=self.feature_to_props(feature, mc))

    ##

    def init_feature(self, req: gws.WebRequester, p: api.InitFeatureRequest) -> gws.Feature:
        """Create a new feature with initial values, without saving it.

        The related features in ``createWithFeatures`` are attached to the new feature.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            The new feature.

        Raises:
            ``gws.ForbiddenError``: If the model is not accessible, not editable or the user cannot create features.
            ``gws.NotFoundError``: If a feature cannot be created from the props.
        """
        mc = self.model_context(req, p, gws.ModelOperation.create)

        f = self.feature_from_props(p.feature, gws.Access.create, mc)
        f.createWithFeatures = [
            self.feature_from_props(r, gws.Access.read, mc)
            for r in (p.feature.createWithFeatures or [])
        ]

        f.model.init_feature(f, mc)
        return f

    def init_feature_response(self, req: gws.WebRequester, p: gws.Request, feature: Optional[gws.Feature]) -> api.InitFeatureResponse:
        """Create the response for ``editInitFeature``.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            feature: Feature to return.

        Returns:
            The response with the feature props.

        Raises:
            ``gws.NotFoundError``: If ``feature`` is ``None``.
        """
        if not feature:
            raise gws.NotFoundError()
        mc = self.model_context(req, p, gws.ModelOperation.create)
        return api.InitFeatureResponse(feature=self.feature_to_props(feature, mc))

    ##

    def write_feature(self, req: gws.WebRequester, p: api.WriteFeatureRequest) -> Optional[gws.Feature]:
        """Validate and save a new or existing feature.

        New features (``isNew``) are created, others are updated. After saving,
        the feature is read back from the model.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            The saved feature as read back, the unsaved feature with ``errors`` set if the
            validation fails, or ``None`` if the saved feature cannot be read back.

        Raises:
            ``gws.ForbiddenError``: If the model is not accessible, not editable or the user lacks permissions.
            ``gws.NotFoundError``: If a feature cannot be created from the props.
        """
        is_new = p.feature.isNew
        mc = self.model_context(req, p, gws.ModelOperation.create if is_new else gws.ModelOperation.update)

        f = self.feature_from_props(p.feature, gws.Access.write, mc)
        f.createWithFeatures = [
            self.feature_from_props(r, gws.Access.read, mc)
            for r in (p.feature.createWithFeatures or [])
        ]

        if not f.model.validate_feature(f, mc):
            return f

        if is_new:
            uid = f.model.create_feature(f, mc)
        else:
            uid = f.model.update_feature(f, mc)

        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        f_created = f.model.get_features([uid], mc)
        if not f_created:
            return

        return f_created[0]

    def write_feature_response(self, req: gws.WebRequester, p: api.WriteFeatureRequest, feature: Optional[gws.Feature]) -> api.WriteFeatureResponse:
        """Create the response for ``editWriteFeature``.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            feature: Saved feature, or a feature with validation errors.

        Returns:
            The response with the feature props, or with the validation errors only.

        Raises:
            ``gws.NotFoundError``: If ``feature`` is ``None``.
        """
        if not feature:
            raise gws.NotFoundError()
        if feature.errors:
            return api.WriteFeatureResponse(validationErrors=feature.errors)

        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        return api.WriteFeatureResponse(
            feature=self.feature_to_props(feature, mc),
            validationErrors=[]
        )

    ##

    def delete_feature(self, req: gws.WebRequester, p: api.DeleteFeatureRequest) -> Optional[gws.Feature]:
        """Delete a feature.

        Args:
            req: Web request.
            p: Request parameters.

        Returns:
            The deleted feature.

        Raises:
            ``gws.ForbiddenError``: If the model is not accessible, not editable or the user cannot delete features.
            ``gws.NotFoundError``: If a feature cannot be created from the props.
        """
        mc = self.model_context(req, p, gws.ModelOperation.delete)
        f = self.feature_from_props(p.feature, gws.Access.delete, mc)
        if f:
            f.model.delete_feature(f, mc)
        return f

    def delete_feature_response(self, req: gws.WebRequester, p: api.DeleteFeatureRequest, feature: Optional[gws.Feature]) -> api.DeleteFeatureResponse:
        """Create the response for ``editDeleteFeature``.

        Args:
            req: Web request.
            p: Request parameters.
            feature: Deleted feature.

        Returns:
            An empty response.
        """
        return api.DeleteFeatureResponse()

    ##

    def require_model(self, model_uid, user: gws.User, access: gws.Access) -> gws.Model:
        """Return an editable model the user can access.

        Args:
            model_uid: Model uid.
            user: User.
            access: Required access.

        Returns:
            The model.

        Raises:
            ``gws.ForbiddenError``: If the model does not exist, is not accessible or is not editable.
        """
        model = cast(gws.Model, user.acquire(model_uid, gws.ext.object.model, access))
        if not model:
            raise gws.ForbiddenError(f'model {model_uid!r} not found or not accessible')
        if not model.isEditable:
            raise gws.ForbiddenError(f'model {model_uid!r} is not editable')
        return model

    def require_field(self, model: gws.Model, field_name: str, user: gws.User, access: gws.Access) -> gws.ModelField:
        """Return a model field the user can access.

        Args:
            model: Model.
            field_name: Field name.
            user: User.
            access: Required access.

        Returns:
            The field.

        Raises:
            ``gws.ForbiddenError``: If the field does not exist or is not accessible.
        """
        field = model.field(field_name)
        if not field:
            raise gws.ForbiddenError(f'field {field_name!r} not found in model {model.uid!r}')
        if not user.can(access, field):
            raise gws.ForbiddenError(f'field {field_name!r} not accessible for user {user.uid!r}')
        return field

    def feature_from_props(self, props: gws.FeatureProps, access: gws.Access, mc: gws.ModelContext) -> gws.Feature:
        """Create a feature from props sent by the client.

        Args:
            props: Feature props, with ``modelUid`` set.
            access: Required access to the model.
            mc: Model context.

        Returns:
            The feature.

        Raises:
            ``gws.ForbiddenError``: If the model is not accessible or not editable.
            ``gws.NotFoundError``: If the model returns no feature for the props.
        """
        model = self.require_model(props.modelUid, mc.user, access)
        feature = model.feature_from_props(props, mc)
        if not feature:
            raise gws.NotFoundError()
        return feature

    def feature_list_to_props(self, features: list[gws.Feature], mc: gws.ModelContext) -> list[gws.FeatureProps]:
        """Convert features to props for the client.

        Renders the list views (``title``, ``label``) of the features and their
        related features and transforms them to the project map CRS.

        Args:
            features: Features to convert.
            mc: Model context.

        Returns:
            A list of feature props.
        """
        template_map = {}

        for f in gws.base.model.iter_features(features, mc):
            if f.model.uid not in template_map:
                template_map[f.model.uid] = gws.u.compact(
                    f.model.root.app.templateMgr.find_template(
                        f'feature.{v}',
                        [f.model, f.model.parent, mc.project], user=mc.user
                    )
                    for v in LIST_VIEWS
                )

            f.render_views(template_map[f.model.uid], user=mc.user, project=mc.project)
            if mc.project and mc.project.map:
                f.transform_to(mc.project.map.bounds.crs)

        return [f.model.feature_to_props(f, mc) for f in features]

    def feature_to_props(self, feature: gws.Feature, mc: gws.ModelContext) -> gws.FeatureProps:
        """Convert a single feature to props for the client.

        Args:
            feature: Feature to convert.
            mc: Model context.

        Returns:
            The feature props.
        """
        ps = self.feature_list_to_props([feature], mc)
        return ps[0]

    def model_context(self, req: gws.WebRequester, p: gws.Request, op, target: Optional[gws.ModelReadTarget] = None, max_depth=1):
        """Create a model context for a request.

        Args:
            req: Web request.
            p: Request parameters, used for the project uid.
            op: Model operation.
            target: Read target.
            max_depth: Maximum depth of related features to process.

        Returns:
            The model context.

        Raises:
            ``gws.NotFoundError``: If the project does not exist.
            ``gws.ForbiddenError``: If the user cannot read the project.
        """
        return gws.ModelContext(
            op=op,
            target=target,
            user=req.user,
            project=req.user.require_project(p.projectUid),
            maxDepth=max_depth,
        )
