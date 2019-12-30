import gws.web
import gws.types as t


class OptionsParams(t.Params):
    pass


class OptionsResponse(t.Response):
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int = 10


class Config(t.WithTypeAndAccess):
    """Dimension action"""
    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


class Object(gws.ActionObject):

    def api_options(self, req: t.IRequest, p: OptionsParams) -> OptionsResponse:
        req.require_project(p.projectUid)

        return OptionsResponse({
            'layerUids': self.var('layers') or [],
            'pixelTolerance': self.var('pixelTolerance'),

        })
