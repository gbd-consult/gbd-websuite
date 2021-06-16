"""Backend for vector edit operations."""

import gws
import gws.base.action
import gws.gis.feature
import gws.lib.json2
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Feature edit action"""
    pass


class EditParams(t.Params):
    layerUid: str
    features: t.List[t.FeatureProps]


class EditResponse(t.Response):
    features: t.List[t.FeatureProps]


class Object(gws.base.action.Object):
    def api_add_features(self, req: t.IRequest, p: EditParams) -> EditResponse:
        """Add features to the layer"""

        return self._handle('insert', req, p)

    def api_delete_features(self, req: t.IRequest, p: EditParams) -> EditResponse:
        """Delete features from the layer"""

        return self._handle('delete', req, p)

    def api_update_features(self, req: t.IRequest, p: EditParams) -> EditResponse:
        """Update features on the layer"""

        return self._handle('update', req, p)

    def _handle(self, op, req, p: EditParams):
        layer: t.ILayer = req.require('gws.ext.layer', p.layerUid)
        if not layer.edit_access(req.user):
            raise gws.web.error.Forbidden()

        features = layer.edit_operation(op, p.features)
        for f in features:
            f.transform_to(layer.map.crs)
            f.apply_templates()
            f.apply_data_model()

        return EditResponse(features=[f.props for f in features])
