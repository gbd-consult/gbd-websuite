"""Backend for vector edit operations."""

import gws
import gws.common.action
import gws.gis.shape
import gws.gis.feature
import gws.tools.json2
import gws.web.error
import gws.common.layer.types
import gws.common.model

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Feature edit action"""
    pass


class GetLayersResponse(t.Response):
    layers: t.List[gws.common.layer.types.LayerProps]


class GetFeaturesParams(t.Params):
    layerUid: str


##


class GetModelsParams(t.Params):
    pass


class GetModelsResponse(t.Response):
    models: t.List[gws.common.model.Props]


##

class ListParams(t.Params):
    features: t.List[t.FeatureProps]


class QueryParams(t.Params):
    layerUids: t.List[str]
    shapes: t.Optional[t.List[t.ShapeProps]]
    keyword: t.Optional[str]
    tolerance: t.Optional[str]
    resolution: float


class ListResponse(t.Response):
    features: t.List[t.FeatureProps]


class FeatureParams(t.Params):
    feature: t.FeatureProps


class FeatureResponse(t.Response):
    feature: t.FeatureProps


_COMBINED_UID_DELIMITER = '___'


class _PreparedCollection(t.Data):
    by_layer: dict
    layers: dict
    features: t.List[t.IFeature]


class Object(gws.common.action.Object):
    def api_get_layers(self, req: t.IRequest, p: t.Params) -> GetLayersResponse:
        project = req.require_project(p.projectUid)
        layers = self._editable_layers(req, project)
        return GetLayersResponse(layers=[la.props for la in layers])

    def api_get_models(self, req: t.IRequest, p: GetModelsParams) -> GetModelsResponse:
        project = req.require_project(p.projectUid)

        models = []

        for la in self._editable_layers(req, project):
            models.append(la.editor.model)

        return GetModelsResponse(models=[m.props_for(req.user) for m in models])

    def api_query_features(self, req: t.IRequest, p: QueryParams) -> ListResponse:
        project = req.require_project(p.projectUid)
        layers = self._editable_layers(req, project)

        if p.layerUids:
            layers = [la for la in layers if la.uid in p.layerUids]

        PIXEL_TOLERANCE = 10

        args = t.SelectArgs(
            keyword=p.keyword,
            depth=1,
            map_tolerance=PIXEL_TOLERANCE * p.resolution)

        if p.shapes:
            args.shape = gws.gis.shape.union([gws.gis.shape.from_props(s) for s in p.shapes])

        out_features = []

        for layer in layers:
            out_features.extend(layer.editor.model.select(args))

        for fe in out_features:
            self._apply_templates_deep(fe, 'title')

        return ListResponse(features=[fe.props for fe in out_features])

    def api_init_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        fe_in = self._load_feature(req, p)

        if not req.user.can_use(fe_in.model.permissions.read):
            raise gws.web.error.Forbidden()

        fe = fe_in
        self._apply_permissions_and_defaults(fe, req, 'read')
        self._apply_templates_deep(fe, 'title')

        return FeatureResponse(feature=fe.props)

    def api_read_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        fe_in = self._load_feature(req, p)

        if not req.user.can_use(fe_in.model.permissions.read):
            raise gws.web.error.Forbidden()

        fe = fe_in.model.get_feature(fe_in.uid)
        if not fe:
            raise gws.web.error.NotFound()

        gws.p(fe.model.uid, list(fe.attributes.keys()))
        self._apply_permissions_and_defaults(fe, req, 'read')
        gws.p(list(fe.attributes.keys()))
        self._apply_templates_deep(fe, 'title')

        return FeatureResponse(feature=fe.props)

    def api_write_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        fe = self._load_feature(req, p)

        if fe.is_new and not req.user.can_use(fe.model.permissions.create):
            raise gws.web.error.Forbidden()
        if not fe.is_new and not req.user.can_use(fe.model.permissions.write):
            raise gws.web.error.Forbidden()

        self._apply_permissions_and_defaults(fe, req, 'write')

        errors = fe.model.validate(fe)
        if errors:
            fe.attributes = {}
            fe.errors = errors
            return FeatureResponse(feature=fe.props)

        fe.model.save(fe)
        gws.common.model.session.commit()
        fe.model.reload(fe)
        fe.is_new = False
        self._apply_templates_deep(fe, 'title')

        return FeatureResponse(feature=fe.props)

    def api_delete_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        fe = self._load_feature(req, p)

        if not req.user.can_use(fe.model.permissions.delete):
            raise gws.web.error.Forbidden()

        fe.model.delete(fe)
        gws.common.model.session.commit()

        return FeatureResponse(feature=None)

    def _apply_permissions_and_defaults(self, fe: t.IFeature, req: t.IRequest, mode):
        fe.model.apply_defaults(fe, mode)
        for f in fe.model.fields:
            if not req.user.can_use(f.permissions.get(mode)):
                gws.log.debug(f'remove field={f.name!r} mode={mode!r}')
                del fe.attributes[f.name]
        return fe

    def _load_feature(self, req: t.IRequest, p: FeatureParams) -> t.IFeature:
        layer_uid = p.feature.layerUid
        layer = t.cast(t.ILayer, self.root.find('gws.ext.layer', layer_uid))
        if not layer or not req.user.can_use(layer.editor):
            raise gws.web.error.Forbidden()
        fe = layer.editor.model.feature_from_props(p.feature, depth=1)
        return fe

    # def _prepare_features(self, req: t.IRequest, p: ListParams) -> t.Optional[_PreparedCollection]:
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
    def _apply_templates_deep(self, fe: t.IFeature, key):
        fe.apply_template(key)
        for k, v in fe.attributes.items():
            if isinstance(v, t.IFeature):
                self._apply_templates_deep(v, key)
            if isinstance(v, list) and v and isinstance(v[0], t.IFeature):
                for f in v:
                    self._apply_templates_deep(f, key)

    def _editable_layers(self, req, project):
        ls = []
        for la in self._enum_layers(project.map):
            if req.user.can_use(la.editor):
                ls.append(la)
        return ls

    def _enum_layers(self, obj):
        layers = getattr(obj, 'layers', None)
        if not layers:
            return [obj]

        ls = []
        for la in layers:
            ls.extend(self._enum_layers(la))
        return ls

    def _list_response(self, features) -> ListResponse:
        return ListResponse(features=[fe.props for fe in features])
