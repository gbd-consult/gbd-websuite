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


class GetPathParams(t.Params):
    layerUid: str
    featureUid: str
    fieldName: str


##

class ListParams(t.Params):
    features: t.List[t.FeatureProps]


class QueryParams(t.Params):
    layerUids: t.List[str]
    shapes: t.Optional[t.List[t.ShapeProps]]
    tolerance: t.Optional[str]
    resolution: float


class ListResponse(t.Response):
    features: t.List[t.FeatureProps]


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

    def http_get_path(self, req: t.IRequest, p: GetPathParams) -> t.FileResponse:
        layer = t.cast(t.ILayer, req.acquire('gws.ext.layer', p.layerUid))
        if not layer or not layer.edit_access(req.user):
            raise gws.web.error.Forbidden()

        with gws.common.model.session():
            args = t.SelectArgs(uids=[p.featureUid])
            flist = layer.editor.model.select(args)

        if not flist:
            raise gws.web.error.NotFound()

        field = layer.editor.model.get_field(p.fieldName)
        if not field or not hasattr(field, 'get_file'):
            raise gws.web.error.NotFound()

        return field.get_file(flist[0])

    def api_get_models(self, req: t.IRequest, p: GetModelsParams) -> GetModelsResponse:
        project = req.require_project(p.projectUid)

        models = []

        for la in self._enum_layers(project.map):
            if la.editor and la.editor.model:
                models.append(la.editor.model.props)

        return GetModelsResponse(models=models)

    def api_init_features(self, req: t.IRequest, p: ListParams) -> ListResponse:
        pc = self._prepare_features(req, p)
        if not pc:
            return self._list_response([])

        for fe in pc.features:
            self._apply_templates_deep(fe, 'title')

        return self._list_response(pc.features)

    def api_query_features(self, req: t.IRequest, p: QueryParams) -> ListResponse:
        project = req.require_project(p.projectUid)
        layers = self._editable_layers(req, project)

        if p.layerUids:
            layers = [la for la in layers if la.uid in p.layerUids]

        out_features = []

        shape = gws.gis.shape.union([gws.gis.shape.from_props(s) for s in p.shapes])

        PIXEL_TOLERANCE = 10

        with gws.common.model.session():
            for layer in layers:
                args = t.SelectArgs(shape=shape, depth=1, map_tolerance=PIXEL_TOLERANCE * p.resolution)
                out_features.extend(layer.editor.model.select(args))

        for fe in out_features:
            self._apply_templates_deep(fe, 'title')

        return self._list_response(out_features)

    def api_read_features(self, req: t.IRequest, p: ListParams) -> ListResponse:
        pc = self._prepare_features(req, p)
        if not pc:
            return self._list_response([])

        out_features = [None] * len(pc.features)

        with gws.common.model.session():
            for layer_uid, indexes in pc.by_layer.items():
                layer = pc.layers[layer_uid]
                args = t.SelectArgs(uids=[pc.features[n].uid for n in indexes], depth=1)
                found = layer.editor.model.select(args)
                by_uid = {str(fe.uid): fe for fe in found}
                for n in indexes:
                    out_features[n] = by_uid.get(str(pc.features[n].uid))

        out_features = gws.compact(out_features)

        for fe in out_features:
            self._apply_templates_deep(fe, 'title')

        return self._list_response(out_features)

    def api_write_features(self, req: t.IRequest, p: ListParams) -> ListResponse:
        """Write features on the layer"""

        pc = self._prepare_features(req, p)
        if not pc:
            return self._list_response([])

        has_validation_errors = False
        for fe in pc.features:
            errors = fe.model.validate(fe)
            if errors:
                fe.errors = errors
                has_validation_errors = True

        if has_validation_errors:
            return self._list_response(pc.features)

        with gws.common.model.session() as sess:
            for layer_uid, indexes in pc.by_layer.items():
                layer = pc.layers[layer_uid]
                for n in indexes:
                    fe = pc.features[n]
                    layer.editor.model.save(fe, fe.is_new)

            sess.commit()

            for fe in pc.features:
                fe.layer.editor.model.reload(fe, depth=1)
                fe.is_new = False
                self._apply_templates_deep(fe, 'title')

        return self._list_response(pc.features)

    def api_delete_features(self, req: t.IRequest, p: ListParams) -> ListResponse:
        """Delete features"""

        pc = self._prepare_features(req, p)
        if not pc:
            return self._list_response([])

        with gws.common.model.session() as sess:
            for layer_uid, indexes in pc.by_layer.items():
                layer = pc.layers[layer_uid]
                for n in indexes:
                    fe = pc.features[n]
                    layer.editor.model.delete(fe)
            sess.commit()

        return self._list_response([])

    def _prepare_features(self, req: t.IRequest, p: ListParams) -> t.Optional[_PreparedCollection]:
        pc = _PreparedCollection(
            by_layer={},
            layers={},
            features=[],
        )

        for index, fp in enumerate(p.features):
            lid = fp.layerUid

            if lid not in pc.by_layer:
                layer = t.cast(t.ILayer, req.acquire('gws.ext.layer', lid))
                if not layer or not layer.edit_access(req.user):
                    raise gws.web.error.Forbidden()
                pc.by_layer[lid] = []
                pc.layers[lid] = layer

            fe = gws.gis.feature.from_props(pc.layers[lid].editor.model, fp, depth=1)
            pc.features.append(fe)
            pc.by_layer[lid].append(index)

        if not pc.features:
            return

        return pc

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
            if la.edit_access(req.user) and la.editor and la.editor.model:
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
