"""Generic group layer."""

import gws
import gws.config
import gws.gis.bounds
import gws.gis.source
import gws.types as t

from . import core

gws.ext.new.layer('group')


class Config(core.Config):
    """Group layer"""

    layers: list[gws.ext.config.layer]
    """layers in this group"""


class Props(core.Props):
    layers: list[gws.ext.props.layer]


class Object(core.Object):

    def configure(self):
        self.configure_group(self.cfg('layers'))

    def props(self, user):
        return gws.merge(super().props(user), layers=self.layers, type='group')
