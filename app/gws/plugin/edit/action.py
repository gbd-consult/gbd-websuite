"""Backend for vector edit operations."""

import gws

import gws.base.action
import gws.base.feature
import gws.base.layer
import gws.base.legend
import gws.base.shape
import gws.base.template
import gws.gis.cache
import gws.base.web.error
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
    resolution: float
    shapes: t.Optional[list[gws.ShapeProps]]
    tolerance: t.Optional[str]


class FeatureRequest(BaseRequest):
    modelUid: str
    feature: gws.FeatureProps


class FeatureResponse(gws.Response):
    feature: gws.FeatureProps


class FeatureListResponse(gws.Response):
    features: list[gws.FeatureProps]


class Object(gws.base.action.Object):

    @gws.ext.command.api('editQueryFeatures')
    def api_query_features(self, req: gws.IWebRequester, p: QueryRequest) -> FeatureListResponse:

        props = []
        project = req.require_project(p.projectUid)

        search = gws.SearchArgs(
            access=gws.Access.write,
            project=project,
            bounds=project.map.bounds,
            tolerance=_DEFAULT_TOLERANCE
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

        for model_uid in p.modelUids:
            model = req.require_model(model_uid)
            if not req.user.can_write(model):
                continue
            features = model.find_features(search, req.user)
            props.extend(self._feature_props(req, p, features))

        return FeatureListResponse(features=props)

    @gws.ext.command.api('editWriteFeature')
    def api_write_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> FeatureResponse:
        model = req.require_model(p.modelUid)
        features = [model.feature_from_props(p.feature, req.user)]
        model.write_features(features, req.user)
        props = self._feature_props(req, p, features)
        return FeatureResponse(feature=props[0])

    @gws.ext.command.api('editInitFeature')
    def api_init_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> FeatureResponse:
        model = req.require_model(p.modelUid)
        if not req.user.can_create(model):
            raise gws.base.web.error.Forbidden()
        features = [model.feature_from_props(p.feature, req.user)]
        props = self._feature_props(req, p, features)
        return FeatureResponse(feature=props[0])

    @gws.ext.command.api('editDeleteFeature')
    def api_delete_feature(self, req: gws.IWebRequester, p: FeatureRequest) -> FeatureResponse:
        model = req.require_model(p.modelUid)
        if not req.user.can_delete(model):
            raise gws.base.web.error.Forbidden()
        features = [model.feature_from_props(p.feature, req.user)]
        model.delete_features(features, req.user)
        return FeatureResponse(feature=None)

    ##

    def _feature_props(self, req: gws.IWebRequester, p: BaseRequest, features: list[gws.IFeature]) -> list[t.Optional[gws.Props]]:
        if not features:
            return []

        project = req.require_project(p.projectUid)
        layer = t.cast(gws.ILayer, req.acquire(p.layerUid, gws.ext.object.layer))
        views = p.views or _DEFAULT_VIEWS

        templates = {}
        props = []

        for feature in features:
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

            props.append(gws.props(feature, req.user, self))

        return props

    # def _apply_permissions_and_defaults(self, fe: t.IFeature, req: gws.IWebRequester, project, mode):
    #     env = t.Data(user=req.user, project=project)
    #     for f in fe.model.fields:
    #         if f.apply_value(fe, mode, 'fixed', env):
    #             continue
    #         if not req.user.can_use(f.permissions.get(mode)):
    #             gws.log.debug(f'remove field={f.name!r} mode={mode!r}')
    #             del fe.attributes[f.name]
    #         f.apply_value(fe, mode, 'default', env)
    #     return fe
    #
    # def _load_feature(self, req: gws.IWebRequester, p: FeatureParams) -> t.IFeature:
    #     layer_uid = p.feature.layerUid
    #     layer = t.cast(t.ILayer, self.root.find('gws.ext.layer', layer_uid))
    #     if not layer or not req.user.can_use(layer.editor):
    #         raise gws.web.error.Forbidden()
    #     fe = layer.editor.model.feature_from_props(p.feature, depth=1)
    #     return fe
    #
    # def _prepare_features(self, req: gws.IWebRequester, p: ListParams) -> t.Optional[_PreparedCollection]:
    #     pc = _PreparedCollection(
    #         by_layer={},
    #         layers={},
    #         features=[],
    #     )
    #
    #     for index, fp in enumerate(p.features):
    #         lid = fp.layerUid
    #
    #         if lid not in pc.by_layer:
    #             layer = t.cast(t.ILayer, self.root.find('gws.ext.layer', lid))
    #             if not layer or not layer.edit_access(req.user):
    #                 raise gws.web.error.Forbidden()
    #             pc.by_layer[lid] = []
    #             pc.layers[lid] = layer
    #
    #         fe = pc.layers[lid].editor.model.feature_from_props(fp, depth=1)
    #         pc.features.append(fe)
    #         pc.by_layer[lid].append(index)
    #
    #     if not pc.features:
    #         return
    #
    #     return pc
    #
    # def _apply_templates_deep(self, fe: t.IFeature, key):
    #     fe.apply_template(key)
    #     for k, v in fe.attributes.items():
    #         if isinstance(v, t.IFeature):
    #             self._apply_templates_deep(v, key)
    #         if isinstance(v, list) and v and isinstance(v[0], t.IFeature):
    #             for f in v:
    #                 self._apply_templates_deep(f, key)
    #
    # def _editable_layers(self, req, project):
    #     ls = []
    #     for la in self._enum_layers(project.map):
    #         if req.user.can_use(la.editor):
    #             ls.append(la)
    #     return ls
    #
    # def _enum_layers(self, obj):
    #     layers = getattr(obj, 'layers', None)
    #     if not layers:
    #         return [obj]
    #
    #     ls = []
    #     for la in layers:
    #         ls.extend(self._enum_layers(la))
    #     return ls
    #
    # def _list_response(self, features) -> FeatureListResponse:
    #     return FeatureListResponse(features=[fe.props for fe in features])
