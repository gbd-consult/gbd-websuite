"""Backend for vector edit operations."""

import gws
import gws.base.action
import gws.base.feature
import gws.base.layer
import gws.base.legend
import gws.base.shape
import gws.base.template
import gws.base.web
import gws.gis.cache
import gws.gis.crs
import gws.gis.render
import gws.lib.image
import gws.lib.jsonx
import gws.lib.mime
import gws.types as t

gws.ext.new.action('edit')

_LIST_VIEWS = ['title', 'label']
_DEFAULT_TOLERANCE = 10, gws.Uom.px


class Config(gws.base.action.Config):
    """Edit action"""
    pass


class Props(gws.base.action.Props):
    pass


##

class GetModelsRequest(gws.Request):
    pass


class GetModelsResponse(gws.Response):
    models: list[gws.ext.props.model]


class GetFeaturesRequest(gws.Request):
    modelUids: list[str]
    crs: t.Optional[gws.CrsName]
    extent: t.Optional[gws.Extent]
    featureUids: t.Optional[list[str]]
    keyword: t.Optional[str]
    resolution: t.Optional[float]
    shapes: t.Optional[list[gws.ShapeProps]]
    tolerance: t.Optional[str]


class GetRelatableFeaturesRequest(gws.Request):
    modelUid: str
    fieldName: str
    extent: t.Optional[gws.Extent]
    keyword: t.Optional[str]


class GetFeatureRequest(gws.Request):
    modelUid: str
    featureUid: str


class InitFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class InitRelatedFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps
    fieldName: str
    relatedModelUid: str


class WriteFeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class FeatureRequest(gws.Request):
    modelUid: str
    feature: gws.FeatureProps


class FeatureResponse(gws.Response):
    feature: gws.FeatureProps


class WriteResponse(gws.Response):
    validationErrors: list[gws.ModelValidationError]
    feature: gws.FeatureProps


class FeatureListResponse(gws.Response):
    features: list[gws.FeatureProps]


class Object(gws.base.action.Object):

    @gws.ext.command.api('editGetModels')
    def get_models(self, req: gws.IWebRequester, p: GetModelsRequest) -> GetModelsResponse:
        project = req.require_project(p.projectUid)
        models = self.root.app.modelMgr.collect_editable(project, req.user)
        return GetModelsResponse(models=[gws.props(m, req.user) for m in models])

    @gws.ext.command.api('editGetFeatures')
    def get_features(self, req: gws.IWebRequester, p: GetFeaturesRequest) -> FeatureListResponse:
        mc = gws.ModelContext(
            mode=gws.ModelMode.edit,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        search = gws.SearchQuery(project=mc.project, tolerance=_DEFAULT_TOLERANCE)
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

        features = []

        for model_uid in p.modelUids:
            model = self.require_model(model_uid, req.user, gws.Access.read)
            features.extend(model.find_features(search, mc))

        propses = self.make_propses(features, mc)
        return FeatureListResponse(features=propses)

    @gws.ext.command.api('editGetRelatableFeatures')
    def get_relatable_features(self, req: gws.IWebRequester, p: GetRelatableFeaturesRequest) -> FeatureListResponse:
        mc = gws.ModelContext(
            mode=gws.ModelMode.edit,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=0,
        )

        model = self.require_model(p.modelUid, req.user, gws.Access.read)
        field = self.require_field(model, p.fieldName, req.user, gws.Access.read)
        search = gws.SearchQuery(keyword=p.keyword)
        features = field.find_relatable_features(search, mc)

        propses = self.make_propses(features, mc)
        return FeatureListResponse(features=propses)

    @gws.ext.command.api('editGetFeature')
    def get_feature(self, req: gws.IWebRequester, p: GetFeatureRequest) -> FeatureResponse:
        mc = gws.ModelContext(
            mode=gws.ModelMode.edit,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        model = self.require_model(p.modelUid, req.user, gws.Access.read)

        features = model.get_features([p.featureUid], mc)
        if not features:
            raise gws.NotFoundError()

        propses = self.make_propses(features, mc)
        return FeatureResponse(feature=propses[0])

    @gws.ext.command.api('editInitFeature')
    def init_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> FeatureResponse:
        mc = gws.ModelContext(
            mode=gws.ModelMode.init,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        model = self.require_model(p.modelUid, req.user, gws.Access.create)
        new_feature = model.new_feature_from_props(p.feature, mc)

        propses = self.make_propses([new_feature], mc)
        return FeatureResponse(feature=propses[0])

    @gws.ext.command.api('editInitRelatedFeature')
    def init_related_feature(self, req: gws.IWebRequester, p: InitRelatedFeatureRequest) -> FeatureResponse:
        mc = gws.ModelContext(
            mode=gws.ModelMode.init,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        model = self.require_model(p.modelUid, req.user, gws.Access.write)
        related_model = self.require_model(p.relatedModelUid, req.user, gws.Access.create)

        features = model.features_from_props([p.feature], mc)
        if not features:
            raise gws.NotFoundError()

        new_feature = model.new_related_feature(p.fieldName, features[0], related_model, mc)

        propses = self.make_propses([new_feature], mc)
        return FeatureResponse(feature=propses[0])

    @gws.ext.command.api('editWriteFeature')
    def write_feature(self, req: gws.IWebRequester, p: WriteFeatureRequest) -> WriteResponse:
        is_new = p.feature.isNew

        mc = gws.ModelContext(
            mode=gws.ModelMode.create if is_new else gws.ModelMode.update,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        model = self.require_model(p.modelUid, req.user, gws.Access.create if is_new else gws.Access.write)
        features = model.features_from_props([p.feature], mc)
        if not features:
            raise gws.NotFoundError()

        if not model.validate_features(features, mc):
            return WriteResponse(validationErrors=features[0].errors)

        if is_new:
            uids = model.create_features(features, mc)
        else:
            uids = model.update_features(features, mc)

        mc.mode = gws.ModelMode.edit
        features = model.get_features(uids, mc)

        propses = self.make_propses(features, mc)

        return WriteResponse(validationErrors=[], feature=propses[0])

    @gws.ext.command.api('editDeleteFeature')
    def delete_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> gws.Response:
        mc = gws.ModelContext(
            mode=gws.ModelMode.delete,
            user=req.user,
            project=req.require_project(p.projectUid),
        )
        model = self.require_model(p.modelUid, req.user, gws.Access.delete)
        features = model.features_from_props([p.feature], mc)
        model.delete_features(features, mc)
        return gws.Response()

    ##

    def require_model(self, model_uid, user: gws.IUser, access: gws.Access) -> gws.IModel:
        model = t.cast(gws.IModel, user.acquire(model_uid, gws.ext.object.model, access))
        if not model or not model.isEditable:
            raise gws.ForbiddenError()
        return model

    def require_field(self, model: gws.IModel, field_name: str, user: gws.IUser, access: gws.Access) -> gws.IModelField:
        field = model.field(field_name)
        if not field or not user.can(access, field):
            raise gws.ForbiddenError()
        return field

    def make_propses(self, features: list[gws.IFeature], mc: gws.ModelContext) -> list[gws.FeatureProps]:
        propses = []
        template_map = {}

        for feature in features:
            self.format_feature(feature, template_map, mc, 1)
            propses.extend(feature.model.features_to_props([feature], mc))

        return propses

    def format_feature(self, feature, template_map, mc, depth):
        model = feature.model

        if model.uid not in template_map:
            template_map[model.uid] = gws.compact(
                self.root.app.templateMgr.locate_template(
                    model, model.parent, mc.project, user=mc.user, subject=f'feature.{v}')
                for v in _LIST_VIEWS
            )

        feature.transform_to(mc.project.map.bounds.crs)
        feature.render_views(template_map[model.uid], user=mc.user, project=mc.project)

        if depth < 1:
            return

        for val in feature.attributes.values():

            if isinstance(val, gws.base.feature.Feature):
                self.format_feature(val, template_map, mc, depth - 1)

            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, gws.base.feature.Feature):
                        self.format_feature(v, template_map, mc, depth - 1)
