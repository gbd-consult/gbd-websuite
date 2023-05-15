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

_DEFAULT_VIEWS = ['title', 'label']
_DEFAULT_TOLERANCE = 10, gws.Uom.px


class Config(gws.base.action.Config):
    """Edit action"""
    pass


class Props(gws.base.action.Props):
    pass


##

class BaseRequest(gws.Request):
    layerUid: t.Optional[str]
    views: t.Optional[list[str]]


class QueryRequest(BaseRequest):
    modelUids: list[str]
    crs: t.Optional[gws.CrsName]
    extent: t.Optional[gws.Extent]
    featureUids: t.Optional[list[str]]
    keyword: t.Optional[str]
    resolution: t.Optional[float]
    relationDepth: t.Optional[int]
    shapes: t.Optional[list[gws.ShapeProps]]
    tolerance: t.Optional[str]


class FeatureRequest(BaseRequest):
    modelUid: str
    feature: gws.FeatureProps


class FeatureResponse(gws.Response):
    feature: gws.FeatureProps


class WriteResponse(gws.Response):
    validationErrors: list[gws.ModelValidationError]


class FeatureListResponse(gws.Response):
    features: list[gws.FeatureProps]


class Object(gws.base.action.Object):

    @gws.ext.command.api('editQueryFeatures')
    def api_query_features(self, req: gws.IWebRequester, p: QueryRequest) -> FeatureListResponse:

        props = []
        project = req.require_project(p.projectUid)

        search = gws.SearchQuery(
            project=project,
            tolerance=_DEFAULT_TOLERANCE,
        )

        if p.extent:
            search.bounds = gws.Bounds(crs=p.crs or project.map.bounds.crs, extent=p.extent)
        if p.shapes:
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]
            search.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])
        if p.resolution:
            search.resolution = p.resolution
        if p.keyword and p.keyword.strip():
            search.keyword = p.keyword.strip()
        if p.featureUids:
            search.uids = p.featureUids
        if p.relationDepth:
            search.relationDepth = p.relationDepth

        for model_uid in p.modelUids:
            model = req.require_model(model_uid)
            features = model.find_features(search, req.user)
            props.extend(self._feature_props(req, p, features))

        return FeatureListResponse(features=props)

    @gws.ext.command.api('editWriteFeature')
    def api_write_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> WriteResponse:
        model = req.require_model(p.modelUid)
        feature = model.feature_from_props(p.feature, req.user, relation_depth=1)
        model.write_feature(feature, req.user)
        return WriteResponse(validationErrors=feature.errors)

    @gws.ext.command.api('editInitFeature')
    def api_init_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> FeatureResponse:
        model = req.require_model(p.modelUid)
        if not req.user.can_create(model):
            raise gws.base.web.error.Forbidden()
        features = [model.feature_from_props(p.feature, req.user, relation_depth=1)]
        props = self._feature_props(req, p, features)
        return FeatureResponse(feature=props[0])

    @gws.ext.command.api('editDeleteFeature')
    def api_delete_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> gws.Response:
        model = req.require_model(p.modelUid)
        if not req.user.can_delete(model):
            raise gws.base.web.error.Forbidden()
        feature = model.feature_from_props(p.feature, req.user)
        model.delete_feature(feature, req.user)
        return gws.Response()

    ##

    def _feature_props(self, req: gws.IWebRequester, p: BaseRequest, features: list[gws.IFeature]) -> list[t.Optional[gws.Props]]:
        if not features:
            return []

        project = req.require_project(p.projectUid)
        views = p.views or _DEFAULT_VIEWS

        templates = {}

        def _prepare(feature: gws.IFeature):
            feature.compute_values(gws.Access.read, req.user)
            feature.transform_to(project.map.bounds.crs)

            if feature.model.uid not in templates:
                templates[feature.model.uid] = []
                for v in views:
                    tpl = gws.base.template.locate(
                        feature.model,
                        feature.model.parent,
                        user=req.user,
                        subject=f'feature.{v}')
                    if tpl:
                        templates[feature.model.uid].append(tpl)

            feature.render_views(
                templates[feature.model.uid],
                user=req.user,
                layer=feature.model.parent)

            for a in feature.attributes.values():
                if isinstance(a, gws.base.feature.Feature):
                    _prepare(a)
                elif isinstance(a, list) and a and isinstance(a[0], gws.base.feature.Feature):
                    for f in a:
                        _prepare(f)

        for f in features:
            _prepare(f)

        return [gws.props(f, req.user, self) for f in features]
