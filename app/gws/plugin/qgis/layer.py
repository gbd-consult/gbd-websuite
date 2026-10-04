"""The ``qgis`` layer."""

from typing import Optional, cast

import gws
import gws.base.layer
import gws.config.util

from . import grabber, provider, flatlayer


@gws.ext.config.layer('qgis')
class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    """Layer group that shows a QGIS project as a tree of its layers."""

    provider: Optional[provider.Config]
    """QGIS project the layers are loaded from."""
    compositeRender: bool = False
    """Render all sublayers as one image instead of one image per layer."""
    sqlFilters: Optional[dict]
    """SQL filters for source layers, passed to QGIS Server."""


@gws.ext.object.layer('qgis')
class Object(gws.base.layer.group.Object):
    """Group layer that shows a QGIS project as a tree of layers.

    The child layers are created from the project layer tree. With
    ``compositeRender``, the layer renders its visible children as one image.
    """

    provider: provider.Object
    """QGIS provider."""
    compositeRender: bool = False
    """Render all visible sublayers as one image."""
    sqlFilters: dict
    """SQL filters passed to the child layers, mapping a source layer name (or ``*``) to a filter expression."""

    def configure(self):
        self.compositeRender = self.cfg('compositeRender', default=False)
        self.sqlFilters = self.cfg('sqlFilters', default={})
        if self.compositeRender:
            self.canRenderBox = True
            self.canRenderTile = True

    def post_configure(self):
        if self.compositeRender:
            gws.base.layer.image.Object.post_configure_grabbers(self)

    def create_cache_name(self, cache):
        return gws.u.sha256([
            self.provider.cache_hash(),
            vars(self.imageFormat),
            list(self.wgsExtent),
            cache.requestBuffer,
            cache.requestTiles,
        ])[: gws.base.layer.image.CACHE_NAME_LENGTH]

    def create_grabber(self, opts):
        return grabber.Object(opts, provider=self.provider, params={})

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_group(self):
        if super().configure_group():
            return True
        gws.base.layer.tree.configure_group_layers_for(
            self,
            self.provider.sourceLayers,
            self.provider.create_leaf_layer_config,
        )
        return True

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.provider.metadata
        return True

    def props(self, user):
        p = super().props(user)
        if not self.compositeRender:
            return p

        def _to_leaf(la):
            pla = la.props(user)
            if not pla:
                return pla
            if pla.get('type') == 'group':
                pla['layers'] = [_to_leaf(la) for la in pla['layers']]
            elif pla.get('type') in ('box', 'tile'):
                pla['type'] = 'compositeLeaf'
            return pla

        p = gws.u.merge(
            p,
            type='compositeBox',
            url=self.url_path_for('box'),
            layers=[_to_leaf(la) for la in p['layers']],
        )
        if self.displayMode == gws.LayerDisplayMode.tile:
            p.type = 'compositeTile'
            p.url = self.url_path_for('tile').replace('/z/', '/compositeLayerUids/{c}/z/')
        return p

    def render_box(self, lri):
        if not self.compositeRender:
            return
        lri.renderParams = self.composite_render_params(lri)
        if not lri.renderParams:
            return
        return gws.base.layer.image.Object.render_box(self, lri)

    def render_tile(self, lri):
        if not self.compositeRender:
            return
        lri.renderParams = self.composite_render_params(lri)
        if not lri.renderParams:
            return
        return gws.base.layer.image.Object.render_tile(self, lri)

    def composite_render_params(self, lri: gws.LayerRenderInput) -> Optional[dict]:
        """Build the GetMap parameters for a composite render.

        The parameters combine ``LAYERS`` and ``FILTER`` of the ``qgisflat``
        descendants listed in ``compositeLayerUids`` of the extra render
        parameters. Layers the user cannot read are skipped.

        Args:
            lri: Render input.

        Returns:
            GetMap parameters, or ``None`` if no layers are requested.
        """
        leaves = dict(lri.extraParams or {}).get('compositeLayerUids', [])
        if not leaves:
            return

        layers = []
        filters = []

        for la in self.find_descendants(gws.ext.object.layer):
            if la.uid not in leaves:
                continue
            if not lri.user.can_read(la):
                gws.log.debug(f'skipping {la.uid=} forbidden')
                continue
            if la.extType != 'qgisflat':
                gws.log.debug(f'skipping {la.uid=} {la.extType=}')
                continue
            p = cast(flatlayer.Object, la).render_params(lri, self.sqlFilters)
            if not p:
                gws.log.debug(f'skipping {la.uid=} no params')
                continue
            layers.extend(reversed(p.get('LAYERS', [])))
            f = p.get('FILTER')
            if f:
                filters.append(f)

        if not layers:
            gws.log.debug(f'no layers')

        params = {}
        params['LAYERS'] = list(reversed(layers))
        if filters:
            params['FILTER'] = ';'.join(filters)

        return params
