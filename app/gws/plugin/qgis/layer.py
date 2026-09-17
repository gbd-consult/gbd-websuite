"""qgis layer.

"qgis" layers display QGIS layers as WebSuite layers, keeping the tree structure.
"""

from typing import Optional, cast

import gws
import gws.base.layer
import gws.config.util

from . import grabber, provider, flatlayer

gws.ext.new.layer('qgis')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    """QGIS Tree layer configuration."""

    provider: Optional[provider.Config]
    """Qgis provider."""
    compositeRender: bool = False
    """If true, the layer will be rendered as a single image."""
    sqlFilters: Optional[dict]
    """Per-layer sql filters."""


class Object(gws.base.layer.group.Object):
    serviceProvider: provider.Object
    compositeRender: bool = False
    sqlFilters: dict

    def configure(self):
        self.compositeRender = self.cfg('compositeRender', default=False)
        self.sqlFilters = self.cfg('sqlFilters', default={})
        if self.compositeRender:
            self.canRenderBox = True

    def create_grabber(self, opts):
        if not self.cfg('compositeRender'):
            return
        return grabber.Object(opts, serviceProvider=self.serviceProvider, params={})

    def configure_group(self):
        gws.config.util.configure_service_provider_for(self, provider.Object)

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.serviceProvider.sourceLayers,
            self.serviceProvider.leaf_config,
        )

        self.configure_group_layers(configs)

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.serviceProvider.metadata
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
            url=self.url_path('box'),
            layers=[_to_leaf(la) for la in p['layers']],
        )
        if self.displayMode == gws.LayerDisplayMode.tile:
            p.type = 'compositeTile'
            p.url = self.url_path('tile').replace('/z/', '/compositeLayerUids/{c}/z/')
        return p

    def render_box(self, lri):
        if self.compositeRender:
            lri.renderParams = self.composite_render_params(lri)
            if not lri.renderParams:
                return
        return super().render_box(lri)

    def render_tile(self, lri):
        if self.compositeRender:
            lri.renderParams = self.composite_render_params(lri)
            if not lri.renderParams:
                return
        return super().render_tile(lri)

    def composite_render_params(self, lri: gws.LayerRenderInput) -> Optional[dict]:
        leaves = dict(lri.extraParams or {}).get('compositeLayerUids', [])
        if not leaves:
            return

        layers = []
        filters = []

        for la in self.descendants():
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
