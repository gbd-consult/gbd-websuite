import gws.gis.feature
import gws.tools.json2

import gws.types as t


class Config(t.WithTypeAndAccess):
    """feature edit action"""
    pass


class EditParams(t.Data):
    layerUid: str
    features: t.List[t.FeatureProps]


class EditResponse(t.Response):
    features: t.List[t.FeatureProps]


def _do_edit(operation, req, p: EditParams):
    layer = req.require('gws.ext.gis.layer', p.layerUid, 'write')
    layer.modify_features(operation, p.features)


class Object(gws.Object):
    def api_add_features(self, req, p: EditParams) -> t.Response:
        """Add features to the layer"""

        _do_edit('add', req, p)
        return t.Response()

    def api_delete_features(self, req, p: EditParams) -> t.Response:
        """Delete features from the layer"""

        _do_edit('delete', req, p)
        return t.Response()

    def api_update_features(self, req, p: EditParams) -> t.Response:
        """Update features on the layer"""

        _do_edit('update', req, p)
        return t.Response()
