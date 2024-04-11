import gws
import gws.base.model
import gws.base.template
import gws.gis.extent
import gws.lib.style
import gws.lib.svg
import gws.types as t

from . import core


class Object(core.Object):
    """Base vector layer"""

    # @TODO rasterize vector layers
    canRenderBox = False
    canRenderXyz = False
    canRenderSvg = True

    geometryType: t.Optional[gws.GeometryType] = None
    geometryCrs: t.Optional[gws.ICrs] = None

    def props(self, user):
        return gws.merge(
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
    #     ts = gws.time_start('render_box:to_png')
    #     img = gws.lib.svg.fragment_to_image(fr, size=view.pxSize, format='png')
    #     gws.time_end(ts)
    #     return img.to_bytes()
    #

    def render_svg_fragment(self, lri: gws.LayerRenderInput):
        bounds = lri.view.bounds
        if lri.view.rotation:
            bounds = gws.Bounds(crs=lri.view.bounds.crs, extent=gws.gis.extent.circumsquare(bounds.extent))

        search = gws.SearchQuery(bounds=bounds)
        features = self.get_features_for_view(search, lri.user)

        if not features:
            gws.log.debug(f'render {self}: no features found')
            return

        gws.time_start('render_svg:to_svg')
        tags = []
        for f in features:
            tags.extend(f.to_svg(lri.view, f.views.get('label', ''), lri.style))
        gws.time_end()

        return tags

    def get_features_for_view(self, search, user, view_names=None):
        model = self.root.app.modelMgr.locate_model(self, user=user, access=gws.Access.read)
        if not model:
            return []

        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.render, user=user)
        features = model.find_features(search, mc)
        if not features:
            return []

        for feature in features:
            if search.bounds:
                feature.transform_to(search.bounds.crs)
            if not feature.category:
                feature.category = self.title

        view_names = view_names or ['label']
        templates = []
        for v in view_names:
            tpl = self.root.app.templateMgr.find_template(
                self, search.project, user=user, subject=f'feature.{v}')
            if tpl:
                templates.append(tpl)

        if templates:
            for feature in features:
                feature.render_views(templates, project=search.project, layer=self, user=user)

        return features
