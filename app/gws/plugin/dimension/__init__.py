"""Provide configuration for the client Dimension module."""

import gws
import gws.base.api
import gws.base.storage
import gws.types as t

STORAGE_CATEGORY = 'Dimension'


@gws.ext.Props('action.dimension')
class Props(gws.base.api.action.Props):
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int


@gws.ext.Config('action.dimension')
class Config(gws.WithAccess):
    """Dimension action"""

    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


@gws.ext.Object('action.dimension')
class Object(gws.base.api.action.Object):

    def props_for(self, user):
        return gws.merge(
            super().props_for(user),
            layerUids=self.var('layers') or [],
            pixelTolerance=self.var('pixelTolerance'),
        )

    @gws.ext.command('api.dimension.storage')
    def storage(self, req: gws.IWebRequest, p: gws.base.storage.Params) -> gws.base.storage.Response:
        helper: gws.base.storage.Object = self.root.application.require_helper('storage')
        return helper.handle_action(req, p, STORAGE_CATEGORY)
