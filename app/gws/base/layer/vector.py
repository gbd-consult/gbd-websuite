import gws
import gws.base.model
import gws.gis.extent
import gws.lib.style
import gws.lib.svg
import gws.types as t

from . import layer, lib


class Config(layer.Config):
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.client  #: layer display mode
    editDataModel: t.Optional[gws.base.model.Config]  #: data model for input data
    editStyle: t.Optional[gws.lib.style.Config]  #: style for features being edited
    loadingStrategy: str = 'all'  #: loading strategy for features ('all', 'bbox')
    style: t.Optional[gws.lib.style.Config]  #: style for features


class Object(layer.Object):
    """Base vector layer"""

    can_render_box = True
    can_render_svg = True
    supports_raster_ows = True
    supports_vector_ows = True

    def props(self, user):
        p = super().props(user)

        return gws.merge(
            p,
            type='vector',
            loadingStrategy=self.var('loadingStrategy'),
            url=self.url_path('features'))

    def render_box(self, view, extra_params=None):
        fr = self.render_svg_fragment(view)
        ts = gws.time_start('render_box:to_png')
        img = gws.lib.svg.fragment_to_image(fr, size=view.size_px, format='png')
        gws.time_end(ts)
        return img.to_bytes()

    def render_svg_fragment(self, view, style=None):
        bounds = view.bounds
        if view.rotation:
            bounds = gws.Bounds(crs=view.bounds.crs, extent=gws.gis.extent.circumsquare(bounds.extent))

        ts = gws.time_start('render_svg:get_features')
        found = self.get_features(bounds)
        gws.time_end(ts)

        ts = gws.time_start('render_svg:convert')
        for f in found:
            f.transform_to(bounds.crs)
            f.apply_templates(subjects=['label'])
        gws.time_end(ts)

        ts = gws.time_start('render_svg:to_svg')
        tags = [tag for f in found for tag in f.to_svg_fragment(view, style or self.style)]
        gws.time_end(ts)

        return tags
