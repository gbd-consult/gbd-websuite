"""Provide configuration for the client Dimension module."""

import gws
import gws.base.action

import gws.types as t


class Props(t.Props):
    type: t.Literal = 'dimensions'
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int


class Config(t.WithTypeAndAccess):
    """Dimension action"""

    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


class Object(gws.base.action.Object):
    @property
    def props(self):
        return Props(
            layerUids=self.var('layers') or [],
            pixelTolerance=self.var('pixelTolerance'),
        )
