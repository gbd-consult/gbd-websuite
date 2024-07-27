"""Backend for vector edit operations."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.feature
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.base.shape
import gws.base.template
import gws.base.web
import gws.gis.crs
import gws.gis.render
import gws.lib.image
import gws.lib.jsonx
import gws.lib.mime

LIST_VIEWS = ['title', 'label']
DEFAULT_TOLERANCE = 10, gws.Uom.px


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


class GetRelatableFeaturesRequest(gws.Request):
    modelUid: str
    fieldName: str
    extent: Optional[gws.Extent]
    keyword: Optional[str]


class GetFeatureRequest(gws.Request):
    modelUid: str
    featureUid: str


class InitFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class WriteFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class DeleteFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class FeatureResponse(gws.Response):
    feature: gws.FeatureProps


class WriteResponse(gws.Response):
    validationErrors: list[gws.ModelValidationError]
    feature: gws.FeatureProps


class FeatureListResponse(gws.Response):
    features: list[gws.FeatureProps]


class Action(gws.base.action.Object):

    def get_models_response(self, req: gws.WebRequester, p: GetModelsRequest) -> GetModelsResponse:
        return GetModelsResponse(
            models=[gws.props_of(m, req.user) for m in self.get_models(req, p)]
        )

    def get_models(self, req: gws.WebRequester, p: GetModelsRequest) -> list[gws.Model]:
        project = req.user.require_project(p.projectUid)
        return self.root.app.modelMgr.editable_models(project, req.user)

    ##

    def get_features_response(self, req: gws.WebRequester, p: GetFeaturesRequest) -> FeatureListResponse:
        fs = self.get_features(req, p)
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList)
        return FeatureListResponse(features=self.feature_list_to_props(fs, mc))

    def get_features(self, req: gws.WebRequester, p: GetFeaturesRequest) -> list[gws.Feature]:
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList)

        search = gws.SearchQuery(project=mc.project, tolerance=DEFAULT_TOLERANCE)
        if p.extent:
            search.bounds = gws.Bounds(crs=p.crs or mc.project.map.bounds.crs, extent=p.extent)
        if p.shapes:
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]
            search.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])
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

    ##

    def get_relatable_features_response(self, req: gws.WebRequester, p: GetRelatableFeaturesRequest) -> FeatureListResponse:
        fs = self.get_relatable_features(req, p)
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList)
        return FeatureListResponse(features=self.feature_list_to_props(fs, mc))

    def get_relatable_features(self, req: gws.WebRequester, p: GetRelatableFeaturesRequest) -> list[gws.Feature]:
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editList, max_depth=0)

        model = self.require_model(p.modelUid, req.user, gws.Access.read)
        field = self.require_field(model, p.fieldName, req.user, gws.Access.read)
        search = gws.SearchQuery(keyword=p.keyword)

        return field.find_relatable_features(search, mc)

    ##

    def get_feature_response(self, req: gws.WebRequester, p: GetFeatureRequest) -> FeatureResponse:
        f = self.get_feature(req, p)
        if not f:
            raise gws.NotFoundError()
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        return FeatureResponse(feature=self.feature_to_props(f, mc))

    def get_feature(self, req: gws.WebRequester, p: GetFeatureRequest) -> Optional[gws.Feature]:
        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)

        model = self.require_model(p.modelUid, req.user, gws.Access.read)

        fs = model.get_features([p.featureUid], mc)
        if fs:
            return fs[0]

    ##

    def init_feature_response(self, req: gws.WebRequester, p: InitFeatureRequest) -> FeatureResponse:
        f = self.init_feature(req, p)
        mc = self.model_context(req, p, gws.ModelOperation.create)
        return FeatureResponse(feature=self.feature_to_props(f, mc))

    def init_feature(self, req: gws.WebRequester, p: InitFeatureRequest) -> gws.Feature:
        mc = self.model_context(req, p, gws.ModelOperation.create)

        f = self.feature_from_props(p.feature, gws.Access.create, mc)
        f.createWithFeatures = [
            self.feature_from_props(r, gws.Access.read, mc)
            for r in (p.feature.createWithFeatures or [])
        ]

        f.model.init_feature(f, mc)
        return f

    ##

    def write_feature_response(self, req: gws.WebRequester, p: WriteFeatureRequest) -> WriteResponse:
        f = self.write_feature(req, p)
        if not f:
            raise gws.NotFoundError()
        if f.errors:
            return WriteResponse(validationErrors=f.errors)

        mc = self.model_context(req, p, gws.ModelOperation.read, gws.ModelReadTarget.editForm)
        return WriteResponse(
            feature=self.feature_to_props(f, mc),
            validationErrors=[]
        )

    def write_feature(self, req: gws.WebRequester, p: WriteFeatureRequest) -> Optional[gws.Feature]:
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

    ##

    def delete_feature_response(self, req: gws.WebRequester, p: DeleteFeatureRequest) -> gws.Response:
        self.delete_feature(req, p)
        return gws.Response()

    def delete_feature(self, req: gws.WebRequester, p: DeleteFeatureRequest):
        mc = self.model_context(req, p, gws.ModelOperation.delete)
        f = self.feature_from_props(p.feature, gws.Access.delete, mc)
        f.model.delete_feature(f, mc)

    ##

    def require_model(self, model_uid, user: gws.User, access: gws.Access) -> gws.Model:
        model = cast(gws.Model, user.acquire(model_uid, gws.ext.object.model, access))
        if not model or not model.isEditable:
            raise gws.ForbiddenError()
        return model

    def require_field(self, model: gws.Model, field_name: str, user: gws.User, access: gws.Access) -> gws.ModelField:
        field = model.field(field_name)
        if not field or not user.can(access, field):
            raise gws.ForbiddenError()
        return field

    def feature_from_props(self, props: gws.FeatureProps, access: gws.Access, mc: gws.ModelContext) -> gws.Feature:
        model = self.require_model(props.modelUid, mc.user, access)
        feature = model.feature_from_props(props, mc)
        if not feature:
            raise gws.NotFoundError()
        return feature

    def feature_list_to_props(self, features: list[gws.Feature], mc: gws.ModelContext) -> list[gws.FeatureProps]:
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
            if mc.project.map:
                f.transform_to(mc.project.map.bounds.crs)

        return [f.model.feature_to_props(f, mc) for f in features]

    def feature_to_props(self, feature: gws.Feature, mc: gws.ModelContext) -> gws.FeatureProps:
        ps = self.feature_list_to_props([feature], mc)
        return ps[0]

    def model_context(self, req, p, op, target: Optional[gws.ModelReadTarget] = None, max_depth=1):
        return gws.ModelContext(
            op=op,
            target=target,
            user=req.user,
            project=req.user.require_project(p.projectUid),
            maxDepth=max_depth,
        )
