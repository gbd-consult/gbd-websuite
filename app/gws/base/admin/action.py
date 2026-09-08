"""Admin action."""

import gws
import gws.base.action
import gws.gis.cache.view

gws.ext.new.action('admin')


class Config(gws.base.action.Config):
    """Admin action configuration."""

    pass


class Props(gws.base.action.Props):
    pass


class ViewCacheRequest(gws.Request):
    path: str = ''


class Object(gws.base.action.Object):
    """Admin action."""

    @gws.ext.command.get('adminViewCache')
    def view_cache(self, req: gws.WebRequester, p: ViewCacheRequest) -> gws.ContentResponse:
        """Display the tile cache viewer."""

        self._ensure_admin(req)
        return gws.gis.cache.view.get_content(self.root, p.path)

    def _ensure_admin(self, req: gws.WebRequester):
        if not req.user.has_role('admin'):
            raise gws.ForbiddenError()
