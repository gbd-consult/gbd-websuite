from typing import Optional

import gws
import gws.base.model
import gws.base.template
import gws.gis.extent
import gws.lib.style
import gws.lib.svg

from . import core


class Object(core.Object):
    """Base vector layer"""

    # @TODO rasterize vector layers
    canRenderBox = False
    canRenderXyz = False
    canRenderSvg = True

    geometryType: Optional[gws.GeometryType] = None
    geometryCrs: Optional[gws.Crs] = None

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            type='vector',
            url=self.url_path('features'),
            geometryType=self.geometryType
        )

    def render(self, lri):
        if lri.type == gws.LayerRenderInputType.svg:
            tags = self.render_svg_fragment(lri)
            if tags:
                return gws.LayerRenderOutput(tags=tags)

        # if lri.type == 'box':

    #     fr = self.render_svg_fragment(view)
    #     ts = gws.debug.time_start('render_box:to_png')
    #     img = gws.lib.svg.fragment_to_image(fr, size=view.pxSize, format='png')
    #     gws.debug.time_end(ts)
    #     return img.to_bytes()
    #

    def render_svg_fragment(self, lri: gws.LayerRenderInput):
        bounds = lri.view.bounds
        if lri.view.rotation:
            bounds = gws.Bounds(crs=lri.view.bounds.crs, extent=gws.gis.extent.circumsquare(bounds.extent))

        search = gws.SearchQuery(bounds=bounds)
        features = self.find_features(search, lri.user)
        if not features:
            gws.log.debug(f'render {self}: no features found')
            return

        # @TODO should pick a project template too, cmp. map/action/get_features
        tpl = self.root.app.templateMgr.find_template(f'feature.label', where=[self], user=lri.user)

        gws.debug.time_start('render_svg:to_svg')
        tags = []

        for feature in features:
            if tpl:
                feature.render_views([tpl], layer=self, user=lri.user)
            tags.extend(feature.to_svg(lri.view, feature.views.get('label', ''), lri.style))

        gws.debug.time_end()

        return tags

    def find_features(self, search, user):
        model = self.root.app.modelMgr.find_model(self, user=user, access=gws.Access.read)
        if not model:
            return []

        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.map, user=user)
        features = model.find_features(search, mc)
        if not features:
            return []

        for feature in features:
            if search.bounds:
                feature.transform_to(search.bounds.crs)
            if not feature.category:
                feature.category = self.title

        return features
