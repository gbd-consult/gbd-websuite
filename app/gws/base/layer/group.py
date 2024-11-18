"""Generic group layer."""

import gws
import gws.config
import gws.gis.bounds
import gws.gis.source

from . import core

gws.ext.new.layer('group')


class Config(core.Config):
    """Group layer"""

    layers: list[gws.ext.config.layer]
    """layers in this group"""


class Props(core.Props):
    layers: list[gws.ext.props.layer]


class Object(core.Object):
    isGroup = True

    def configure(self):
        self.configure_group()
        if not self.layers:
            raise gws.Error(f'group {self} is empty')
        self.configure_layer()

    def configure_group(self):
        p = self.cfg('layers')
        if p:
            self.configure_group_layers(p)
            return True

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        self.bounds = gws.gis.bounds.union([la.bounds for la in self.layers])
        return True

    def configure_zoom_bounds(self):
        if super().configure_zoom_bounds():
            return True
        self.zoomBounds = gws.gis.bounds.union([(la.zoomBounds or la.bounds) for la in self.layers])
        return True

    def configure_resolutions(self):
        if super().configure_resolutions():
            return True
        res = set()
        for la in self.layers:
            res.update(la.resolutions)
        self.resolutions = sorted(res)
        return True

    def configure_legend(self):
        if super().configure_legend():
            return True
        layers_uids = [la.uid for la in self.layers if la.legend]
        if layers_uids:
            self.legend = self.create_child(gws.ext.object.legend, type='combined', layerUids=layers_uids)
            return True

    def post_configure(self):
        self.canRenderBox = any(la.canRenderBox for la in self.layers)
        self.canRenderXyz = any(la.canRenderXyz for la in self.layers)
        self.canRenderSvg = any(la.canRenderSvg for la in self.layers)

        self.isSearchable = any(la.isSearchable for la in self.layers)

    ##

    def props(self, user):
        return gws.u.merge(super().props(user), type='group')
