"""Provide configuration for the client Dimension module."""

import gws
import gws.base.action
import gws.base.storage
import gws.types as t

STORAGE_CATEGORY = 'Dimension'


@gws.ext.props.action('dimension')
class Props(gws.base.action.Props):
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int


@gws.ext.config.action('dimension')
class Config(gws.ConfigWithAccess):
    """Dimension action"""

    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


@gws.ext.object.action('dimension')
class Object(gws.base.action.Object):

    def props(self, user):
        return gws.merge(
            super().props(user),
            layerUids=self.var('layers') or [],
            pixelTolerance=self.var('pixelTolerance'),
        )

    @gws.ext.command.api('dimensionStorage')
    def storage(self, req: gws.IWebRequester, p: gws.base.storage.Params) -> gws.base.storage.Response:
        helper: gws.base.storage.Object = self.root.app.require_helper('storage')
        return helper.handle_action(req, p, STORAGE_CATEGORY)
