"""Generic group layer."""

import gws
import gws.config
import gws.gis.bounds
import gws.gis.source
import gws.types as t

from . import core, configure


@gws.ext.config.layer('group')
class Config(core.Config):
    """Group layer"""

    layers: t.List[gws.ext.config.layer] 
    """layers in this group"""


@gws.ext.object.layer('group')
class Object(core.Object):

    def configure(self):
        self.configure_group(self.var('layers'))

    def props(self, user):
        return gws.merge(super().props(user), type='group')
