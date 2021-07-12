"""Backend for vector edit operations."""

import gws
import gws.types as t
import gws.base.api
import gws.lib.feature
import gws.lib.json2
import gws.base.web.error


class Config(gws.WithAccess):
    """Feature edit action"""
    pass


class EditParams(gws.Params):
    layerUid: str
    features: t.List[gws.lib.feature.Props]


class EditResponse(gws.Response):
    features: t.List[gws.lib.feature.Props]


class Object(gws.base.api.Action):
    def api_add_features(self, req: gws.IWebRequest, p: EditParams) -> EditResponse:
        """Add features to the layer"""

        return self._handle('insert', req, p)

    def api_delete_features(self, req: gws.IWebRequest, p: EditParams) -> EditResponse:
        """Delete features from the layer"""

        return self._handle('delete', req, p)

    def api_update_features(self, req: gws.IWebRequest, p: EditParams) -> EditResponse:
        """Update features on the layer"""

        return self._handle('update', req, p)

    def _handle(self, op, req, p: EditParams):
        layer: gws.ILayer = req.require('gws.ext.layer', p.layerUid)
        if not layer.edit_access(req.user):
            raise gws.base.web.error.Forbidden()

        features = layer.edit_operation(op, p.features)
        for f in features:
            f.transform_to(layer.map.crs)
            f.apply_templates()
            f.apply_data_model()

        return EditResponse(features=[f.props for f in features])
