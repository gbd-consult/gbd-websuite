"""Provide configuration for the client Dimension module."""

import gws
import gws.base.api
import gws.base.storage
import gws.types as t

STORAGE_CATEGORY = 'Annotate'


@gws.ext.props.action('annotate')
class Props(gws.base.api.action.Props):
    pass

@gws.ext.config.action('annotate')
class Config(gws.ConfigWithAccess):
    """Annotate action"""

    pass


@gws.ext.object.action('annotate')
class Object(gws.base.api.action.Object):

    @gws.ext.command.api('annotateStorage')
    def storage(self, req: gws.IWebRequest, p: gws.base.storage.Params) -> gws.base.storage.Response:
        helper: gws.base.storage.Object = self.root.application.require_helper('storage')
        return helper.handle_action(req, p, STORAGE_CATEGORY)
