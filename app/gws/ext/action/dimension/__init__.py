import gws.web
import gws.types as t


class Props(t.Props):
    type: t.Literal = 'dimensions'
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int


class Config(t.WithTypeAndAccess):
    """Dimension action"""

    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


class Object(gws.ActionObject):
    @property
    def props(self):
        return Props(
            layerUids=self.var('layers') or [],
            pixelTolerance=self.var('pixelTolerance'),
        )
