"""Base vector layer."""

from typing import Optional

import gws
import gws.base.model
import gws.base.template
import gws.lib.extent
import gws.lib.style
import gws.lib.svg

from . import core


class Object(core.Object):
    """Base vector layer.

    Finds features through the layer's models and renders them as SVG. The
    client loads the features from the ``mapGetFeatures`` command.
    """

    # @TODO rasterize vector layers
    canRenderBox = False
    canRenderTile = False
    canRenderSvg = True

    geometryType: Optional[gws.GeometryType] = None
    """Geometry type of the layer's features."""
    geometryCrs: Optional[gws.Crs] = None
    """CRS of the layer's feature geometries."""

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            type='vector',
            url=self.url_path_for('features'),
            geometryType=self.geometryType
        )

    def render_svg(self, lri):
        tags = self.render_svg_fragment(lri)
        if tags:
            return gws.LayerRenderOutput(tags=tags)
    #

    def render_svg_fragment(self, lri: gws.LayerRenderInput):
        """Render the features in the map view as SVG elements.

        Labels are rendered from the ``feature.label`` template. For a rotated
        view, features are searched in the circumscribed square of the view
        extent.

        Args:
            lri: Render input with the map view, style and user.

        Returns:
            A list of SVG elements, or ``None`` if no features are found.
        """
        bounds = lri.view.bounds
        if lri.view.rotation:
            bounds = gws.Bounds(crs=lri.view.bounds.crs, extent=gws.lib.extent.circumsquare(bounds.extent))

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
