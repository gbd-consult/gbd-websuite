"""Provide configuration for the client Dimension module."""

import gws
import gws.base.action
import gws.base.storage
import gws.types as t

STORAGE_CATEGORY = 'Annotate'


@gws.ext.props.action('annotate')
class Props(gws.base.action.Props):
    pass

@gws.ext.config.action('annotate')
class Config(gws.ConfigWithAccess):
    """Annotate action"""

    pass


@gws.ext.object.action('annotate')
class Object(gws.base.action.Object):

    @gws.ext.command.api('annotateStorage')
    def storage(self, req: gws.IWebRequester, p: gws.base.storage.Params) -> gws.base.storage.Response:
        helper: gws.base.storage.Object = self.root.app.require_helper('storage')
        return helper.handle_action(req, p, STORAGE_CATEGORY)
