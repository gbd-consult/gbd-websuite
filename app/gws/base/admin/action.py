"""Admin action."""

import gws
import gws.base.action

from . import inspector, mapcache


@gws.ext.config.action('admin')
class Config(gws.base.action.Config):
    """Admin action configuration. (added in 8.5)"""

    pass


@gws.ext.props.action('admin')
class Props(gws.base.action.Props):
    pass


class ViewCacheRequest(gws.Request):
    path: str = ''


class InspectorRequest(gws.Request):
    path: str = ''
    search: str = ''


@gws.ext.object.action('admin')
class Object(gws.base.action.Object):
    """Admin action."""

    @gws.ext.command.get('adminMapCache')
    def admin_map_cache(self, req: gws.WebRequester, p: ViewCacheRequest) -> gws.ContentResponse:
        """Display the tile cache viewer."""

        self._ensure_admin(req)
        return mapcache.get_content(self.root, p.path)

    @gws.ext.command.get('adminInspector')
    def admin_inspector(self, req: gws.WebRequester, p: InspectorRequest) -> gws.ContentResponse:
        """Display the object inspector."""

        self._ensure_admin(req)
        return inspector.get_content(self.root, p.path, p.search)

    def _ensure_admin(self, req: gws.WebRequester):
        if not req.user.has_role('admin'):
            raise gws.ForbiddenError()
