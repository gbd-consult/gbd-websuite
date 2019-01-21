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


class Object(gws.Object):
    def api_add_features(self, req, p: EditParams) -> EditResponse:
        """Add features to the layer"""

        layer = req.require('gws.ext.gis.layer', p.layerUid, 'write')
        fs = layer.add_features(p.features)
        return EditResponse({'features': [f.props for f in fs]})

    def api_delete_features(self, req, p: EditParams) -> EditResponse:
        """Delete features from the layer"""

        layer = req.require('gws.ext.gis.layer', p.layerUid, 'write')
        fs = layer.delete_features(p.features)
        return EditResponse({'features': [f.props for f in fs]})

    def api_update_features(self, req, p: EditParams) -> EditResponse:
        """Update features on the layer"""

        layer = req.require('gws.ext.gis.layer', p.layerUid, 'write')
        fs = layer.update_features(p.features)
        return EditResponse({'features': [f.props for f in fs]})
