import gws
import gws.base.api
import gws.base.auth
import gws.lib.intl
import gws.types as t
from . import core


class InfoResponse(gws.Response):
    project: core.Props
    locale: gws.lib.intl.Locale
    user: t.Optional[gws.base.auth.UserProps]


@gws.ext.Object('action.project')
class Action(gws.base.api.action.Object):
    """Project information action"""

    @gws.ext.command('api.project.info')
    def info(self, req: gws.IWebRequest, p: gws.Params) -> InfoResponse:
        """Return the project configuration"""

        project = req.require_project(p.projectUid)

        locale_uid = p.localeUid
        if locale_uid not in project.locale_uids:
            locale_uid = project.locale_uids[0]

        return InfoResponse(
            project=project.props_for(req.user),
            locale=gws.lib.intl.locale(locale_uid),
            user=None if req.user.is_guest else req.user.props)
