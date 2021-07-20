"""Provide configuration for the client Dimension module."""

import gws
import gws.types as t
import gws.base.api


@gws.ext.Props('action.dimension')
class Props(gws.base.api.ActionProps):
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int


@gws.ext.Config('action.dimension')
class Config(gws.WithAccess):
    """Dimension action"""

    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


@gws.ext.Object('action.dimension')
class Object(gws.base.api.Action):
    @property
    def props(self):
        return Props(
            layerUids=self.var('layers') or [],
            pixelTolerance=self.var('pixelTolerance'),
        )
