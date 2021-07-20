import gws
import gws.types as t
import gws.base.map.action
import gws.base.model
import gws.core.debug
import gws.lib.extent
import gws.lib.style
import gws.lib.svg

from . import core

_FEATURE_FULL_FORMAT_THRESHOLD = 500


class Config(core.Config):
    display: core.DisplayMode = core.DisplayMode.client  #: layer display mode
    editDataModel: t.Optional[gws.base.model.Config]  #: data model for input data
    editStyle: t.Optional[gws.lib.style.Config]  #: style for features being edited
    loadingStrategy: str = 'all'  #: loading strategy for features ('all', 'bbox')
    style: t.Optional[gws.lib.style.Config]  #: style for features


class Object(core.Object):
    """Base vector layer"""

    @property
    def props(self):
        p = super().props

        if self.display == 'box':
            return gws.merge(p, {
                'type': 'box',
                'url': gws.base.map.action.url_for_render_box(self.uid),
            })

        return gws.merge(p, {
            'type': 'vector',
            'loadingStrategy': self.var('loadingStrategy'),
            'style': self.style,
            'editStyle': self.edit_style,
            'url': gws.base.map.action.url_for_get_features(self.uid),
        })

    def configure(self):
        self.can_render_box = True
        self.can_render_svg = True
        self.supports_wms = True
        self.supports_wfs = True

    def connect_feature(self, feature: gws.IFeature) -> gws.IFeature:
        feature.layer = self
        feature.templates = self.templates
        feature.data_model = self.data_model
        return feature

    def render_box(self, rv, extra_params=None):
        tags = self.render_svg_tags(rv)
        ts = gws.time_start('render_box:to_png')
        png = gws.lib.svg.as_png(tags, size=rv.size_px)
        gws.time_end(ts)
        return png

    def render_svg_tags(self, rv, style=None):
        bounds = rv.bounds
        if rv.rotation:
            bounds = gws.Bounds(crs=bounds.crs, extent=gws.lib.extent.circumsquare(bounds.extent))

        ts = gws.time_start('render_svg:get_features')
        found = self.get_features(bounds)
        gws.time_end(ts)

        ts = gws.time_start('render_svg:convert')
        for f in found:
            f.transform_to(rv.bounds.crs)
            f.apply_templates(keys=['label'])
        gws.time_end(ts)

        ts = gws.time_start('render_svg:to_svg')
        tags = [tag for f in found for tag in f.to_svg_tags(rv, style or self.style)]
        ts = gws.time_end(ts)

        return tags
