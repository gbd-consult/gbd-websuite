"""qgis layer.

"qgis" layers display QGIS layers as WebSuite layers, keeping the tree structure.
"""

from typing import Optional, cast

import gws
import gws.base.layer
import gws.config.util

from . import provider, flatlayer

gws.ext.new.layer('qgis')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    """QGIS Tree layer configuration."""

    provider: Optional[provider.Config]
    """Qgis provider."""
    compositeRender: bool = False
    """If true, the layer will be rendered as a single image. (added in 8.2)"""
    sqlFilters: Optional[dict]
    """Per-layer sql filters. (added in 8.2)"""


class Object(gws.base.layer.group.Object):
    serviceProvider: provider.Object
    compositeRender: bool = False
    sqlFilters: dict

    def configure(self):
        self.compositeRender = self.cfg('compositeRender', default=False)
        self.sqlFilters = self.cfg('sqlFilters', default={})

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

    def post_configure(self):
        if self.compositeRender:
            self.canRenderBox = True

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
            elif pla.get('type') == 'box':
                pla['type'] = 'compositeLeaf'
            return pla

        return gws.u.merge(
            p,
            type='compositeTree',
            layers=[_to_leaf(la) for la in p['layers']],
        )

    def render(self, lri):
        if not self.compositeRender:
            return super().render(lri)

        if lri.type != gws.LayerRenderInputType.box:
            return

        params = self.get_render_params(lri)
        if not params:
            return

        def get_box(bounds, width, height):
            return self.serviceProvider.get_map(self, bounds, width, height, params)

        content = gws.base.layer.util.generic_render_box(self, lri, get_box)
        return gws.LayerRenderOutput(content=content)

    def get_render_params(self, lri: gws.LayerRenderInput) -> Optional[dict]:
        leaves = dict(lri.extraParams or {}).get('compositeLayerUids', [])
        if not leaves:
            return

        layers = []
        filters = []

        for la in self.descendants():
            if la.uid not in leaves:
                continue
            if la.extType != 'qgisflat':
                gws.log.debug(f'skipping {la.uid=} {la.extType=}')
                continue
            p = cast(flatlayer.Object, la).get_render_params(lri, self.sqlFilters)
            if not p:
                gws.log.debug(f'skipping {la.uid=} no params')
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
