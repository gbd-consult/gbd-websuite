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


class Object(gws.base.action.Object):

    @gws.ext.command.api('editGetModels')
    def get_models(self, req: gws.IWebRequester, p: GetModelsRequest) -> GetModelsResponse:
        project = req.require_project(p.projectUid)
        models = self.root.app.modelMgr.editable_models(project, req.user)
        return GetModelsResponse(models=[gws.props(m, req.user) for m in models])

    @gws.ext.command.api('editGetFeatures')
    def get_features(self, req: gws.IWebRequester, p: GetFeaturesRequest) -> FeatureListResponse:
        mc = gws.ModelContext(
            op=gws.ModelOperation.read,
            readMode=gws.ModelReadMode.list,
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
            op=gws.ModelOperation.read,
            readMode=gws.ModelReadMode.list,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=0,
        )

        model = self.require_model(p.modelUid, req.user, gws.Access.read)
        field = self.require_field(model, p.fieldName, req.user, gws.Access.read)
        search = gws.SearchQuery(keyword=p.keyword)

        fn = getattr(field, 'find_relatable_features', None)
        if not fn:
            raise gws.NotFoundError()

        features = fn(search, mc)
        propses = self.make_propses(features, mc)
        return FeatureListResponse(features=propses)

    @gws.ext.command.api('editGetFeature')
    def get_feature(self, req: gws.IWebRequester, p: GetFeatureRequest) -> FeatureResponse:
        mc = gws.ModelContext(
            op=gws.ModelOperation.read,
            readMode=gws.ModelReadMode.form,
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
    def init_feature(self, req: gws.IWebRequester, p: InitFeatureRequest) -> FeatureResponse:
        mc = gws.ModelContext(
            op=gws.ModelOperation.create,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        feature = self.from_props(p.feature, gws.Access.create, mc)
        feature.createWithFeatures = [
            self.from_props(r, gws.Access.read, mc)
            for r in (p.feature.createWithFeatures or [])
        ]

        feature.model.init_feature(feature, mc)

        propses = self.make_propses([feature], mc)
        return FeatureResponse(feature=propses[0])

    @gws.ext.command.api('editWriteFeature')
    def write_feature(self, req: gws.IWebRequester, p: WriteFeatureRequest) -> WriteResponse:
        is_new = p.feature.isNew

        mc = gws.ModelContext(
            op=gws.ModelOperation.create if is_new else gws.ModelOperation.update,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )

        feature = self.from_props(p.feature, gws.Access.write, mc)
        feature.createWithFeatures = [
            self.from_props(r, gws.Access.read, mc)
            for r in (p.feature.createWithFeatures or [])
        ]

        if not feature.model.validate_feature(feature, mc):
            return WriteResponse(validationErrors=feature.errors)

        if is_new:
            uid = feature.model.create_feature(feature, mc)
        else:
            uid = feature.model.update_feature(feature, mc)

        mc = gws.ModelContext(
            op=gws.ModelOperation.read,
            readMode=gws.ModelReadMode.form,
            user=req.user,
            project=req.require_project(p.projectUid),
            maxDepth=1,
        )
        features = feature.model.get_features([uid], mc)
        if not features:
            raise gws.NotFoundError()

        propses = self.make_propses(features, mc)
        return WriteResponse(validationErrors=[], feature=propses[0])

    @gws.ext.command.api('editDeleteFeature')
    def delete_feature(self, req: gws.IWebRequester, p: DeleteFeatureRequest) -> gws.Response:
        mc = gws.ModelContext(
            op=gws.ModelOperation.delete,
            user=req.user,
            project=req.require_project(p.projectUid),
        )

        feature = self.from_props(p.feature, gws.Access.delete, mc)
        feature.model.delete_feature(feature, mc)
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

    def from_props(self, props: gws.FeatureProps, access: gws.Access, mc: gws.ModelContext):
        model = self.require_model(props.modelUid, mc.user, access)
        feature = model.feature_from_props(props, mc)
        if not feature:
            raise gws.NotFoundError()
        return feature

    def make_propses(self, features: list[gws.IFeature], mc: gws.ModelContext) -> list[gws.FeatureProps]:
        propses = []
        template_map = {}

        gws.time_start('format')
        for feature in features:
            self.format_feature(feature, template_map, mc, 1)
        gws.time_end()

        gws.time_start('ps')
        for feature in features:
            propses.append(feature.model.feature_to_props(feature, mc))
        gws.time_end()

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
