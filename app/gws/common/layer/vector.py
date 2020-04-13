import gws
import gws.common.model
import gws.common.style
import gws.tools.svg
import gws.gis.extent

import gws.types as t

from . import layer, types

_FEATURE_FULL_FORMAT_THRESHOLD = 500


class Config(layer.Config):
    display: types.DisplayMode = 'client'  #: layer display mode
    editDataModel: t.Optional[gws.common.model.Config]  #: data model for input data
    editStyle: t.Optional[gws.common.style.Config]  #: style for features being edited
    loadingStrategy: str = 'all'  #: loading strategy for features ('all', 'bbox')
    style: t.Optional[gws.common.style.Config]  #: style for features


#:export IVectorLayer
class Vector(layer.Layer, t.IVectorLayer):
    def configure(self):
        super().configure()

        self.can_render_box = True
        self.can_render_svg = True
        self.supports_wms = True
        self.supports_wfs = True

    @property
    def props(self):
        p = super().props

        if self.display == 'box':
            return gws.merge(p, {
                'type': 'box',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBox/layerUid/' + self.uid,
            })

        return gws.merge(p, {
            'type': 'vector',
            'loadingStrategy': self.var('loadingStrategy'),
            'style': self.style,
            'editStyle': self.edit_style,
            'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetFeatures/layerUid/' + self.uid,
        })

    def connect_feature(self, feature: t.IFeature) -> t.IFeature:
        feature.layer = self
        return feature

    def render_box(self, rv, extra_params=None):
        tags = self.render_svg_tags(rv)
        gws.debug.time_start('render_box:to_png')
        png = gws.tools.svg.as_png(tags, size=rv.size_px)
        gws.debug.time_start('render_box:to_png')
        return png

    def render_svg_tags(self, rv, style=None):
        bounds = rv.bounds
        if rv.rotation:
            bounds = t.Bounds(crs=bounds.crs, extent=gws.gis.extent.circumsquare(bounds.extent))

        gws.debug.time_start('render_svg:get_features')
        found = self.get_features(bounds)
        gws.debug.time_end('render_svg:get_features')

        gws.debug.time_start('render_svg:convert')
        features = [f.transform_to(rv.bounds.crs).apply_converter() for f in found]
        gws.debug.time_end('render_svg:convert')

        gws.debug.time_start('render_svg:to_svg')
        tags = [tag for f in features for tag in f.to_svg_tags(rv, style or self.style)]
        gws.debug.time_end('render_svg:to_svg')

        return tags
