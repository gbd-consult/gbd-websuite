import gws
import gws.common.model
import gws.common.style
import gws.gis.svg

import gws.types as t

from . import layer, types


class Config(layer.Config):
    display: types.DisplayMode = 'client'  #: layer display mode
    editDataModel: t.Optional[gws.common.model.Config]  #: data model for input data
    editStyle: t.Optional[gws.common.style.Config]  #: style for features being edited
    loadingStrategy: str = 'all'  #: loading strategy for features ('all', 'bbox')
    style: t.Optional[gws.common.style.Config]  #: style for features


#:export IVectorLayer
class Vector(layer.Layer, t.IVectorLayer):
    def __init__(self):
        super().__init__()

        self.can_render_box = True
        self.can_render_svg = True
        self.supports_wms = True
        self.supports_wfs = True

    @property
    def props(self):
        if self.display == 'box':
            return super().props.extend({
                'type': 'box',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBox/layerUid/' + self.uid,
            })

        return super().props.extend({
            'type': 'vector',
            'loadingStrategy': self.var('loadingStrategy'),
            'style': self.style,
            'editStyle': self.edit_style,
            'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetFeatures/layerUid/' + self.uid,
        })

    def connect_feature(self, feature: t.IFeature) -> t.IFeature:
        feature.layer = self
        return feature

    def render_box(self, rv, client_params=None):
        elements = self.render_svg(rv)
        return gws.gis.svg.to_png(elements, size=rv.size_px)

    def render_svg(self, rv, style=None):
        features = self.get_features(rv.bounds)
        for f in features:
            f.convert(target_crs=rv.bounds.crs)
        return [f.to_svg(rv, style or self.style) for f in features]
