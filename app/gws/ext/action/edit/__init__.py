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
            m = la.editor.model
            p = m.props_for(req.user)
            p['layerUid'] = la.uid
            models.append(p)

        return GetModelsResponse(models=models)

    def api_query_features(self, req: t.IRequest, p: QueryParams) -> ListResponse:
        project = req.require_project(p.projectUid)
        layers = self._editable_layers(req, project)

        if p.layerUids:
            layers = [la for la in layers if la.uid in p.layerUids]

        PIXEL_TOLERANCE = 10

        args = t.SelectArgs(
            keyword=p.keyword,
            depth=1,
            map_tolerance=PIXEL_TOLERANCE * p.resolution,
            project=project,
            user=req.user,
        )

        if p.shapes:
            args.shape = gws.gis.shape.union([gws.gis.shape.from_props(s) for s in p.shapes])

        out_features = []
        mc = t.ModelContext(access='read', user=req.user, project=project, depth=1)

        for layer in layers:
            args.extra_where = []
            if layer.table:
                flt = project.variable('table_filters', {}).get(layer.table.name)
                if flt:
                    gws.log.debug(f'using {flt=} for {layer.uid=}')
                    args.extra_where = [flt]

            out_features.extend(layer.editor.model.select(args, mc))

        for fe in out_features:
            self._apply_templates_deep(fe, 'title')

        return ListResponse(features=[self._feature_props(fe, mc) for fe in out_features])

    def api_init_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        project = req.require_project(p.projectUid)
        mc = t.ModelContext(access='read', user=req.user, project=project, depth=0)
        fe_in = self._load_feature(p, mc)

        if not req.user.can_use(fe_in.model.permissions.read):
            raise gws.web.error.Forbidden()

        fe = fe_in
        fe.model.apply_permissions_and_defaults(fe, mc)
        self._apply_templates_deep(fe, 'title')

        return FeatureResponse(feature=self._feature_props(fe, mc))

    def api_read_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        project = req.require_project(p.projectUid)
        mc = t.ModelContext(access='read', user=req.user, project=project, depth=1)
        fe_in = self._load_feature(p, mc)

        if not req.user.can_use(fe_in.model.permissions.read):
            raise gws.web.error.Forbidden()

        fe = fe_in.model.get_feature(fe_in.uid, mc)
        if not fe:
            raise gws.web.error.NotFound()

        fe.model.apply_permissions_and_defaults(fe, mc)
        self._apply_templates_deep(fe, 'title')

        return FeatureResponse(feature=self._feature_props(fe, mc))

    def api_write_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        project = req.require_project(p.projectUid)
        mc = t.ModelContext(access='write', user=req.user, project=project, depth=1)
        fe = self._load_feature(p, mc)

        if fe.is_new and not req.user.can_use(fe.model.permissions.create):
            raise gws.web.error.Forbidden()
        if not fe.is_new and not req.user.can_use(fe.model.permissions.write):
            raise gws.web.error.Forbidden()

        fe.model.apply_permissions_and_defaults(fe, mc)

        errors = fe.model.validate(fe, mc)
        if errors:
            fe.attributes = {}
            fe.errors = errors
            return FeatureResponse(feature=self._feature_props(fe, mc))

        try:
            fe = fe.model.save(fe, mc)
            gws.common.model.session.commit()
        except Exception as exc:
            gws.log.exception()
            try:
                msg = exc.args[0]
            except:
                msg = None
            raise gws.web.error.BadRequest(msg)

        self._apply_templates_deep(fe, 'title')

        return FeatureResponse(feature=self._feature_props(fe, mc))

    def api_delete_feature(self, req: t.IRequest, p: FeatureParams) -> FeatureResponse:
        project = req.require_project(p.projectUid)
        mc = t.ModelContext(access='write', user=req.user, project=project, depth=1)
        fe = self._load_feature(p, mc)

        if not req.user.can_use(fe.model.permissions.delete):
            raise gws.web.error.Forbidden()

        fe.model.delete(fe, mc)
        gws.common.model.session.commit()

        return FeatureResponse(feature=None)

    def _load_feature(self, p: FeatureParams, mc: t.ModelContext) -> t.IFeature:
        layer_uid = p.feature.layerUid
        layer = t.cast(t.ILayer, self.root.find('gws.ext.layer', layer_uid))
        if not layer or not mc.user.can_use(layer.editor):
            raise gws.web.error.Forbidden()
        fe = layer.editor.model.feature_from_props(p.feature, mc)
        return fe

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
            if req.user.can_use(la.editor, parent=project):
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

    def _feature_props(self, fe: t.IFeature, mc: t.ModelContext):
        p = fe.model.feature_props(fe, mc)
        p.layerUid = mc.project.uid + '.map.' + p.layerUid.split('.')[-1]
        return p
