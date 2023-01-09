import gws
import gws.base.action
import gws.base.auth.user
import gws.lib.intl
import gws.types as t

from . import core


class InfoResponse(gws.Response):
    project: gws.ext.props.project
    locale: gws.Locale
    user: t.Optional[gws.base.auth.user.Props]


@gws.ext.object.action('project')
class Object(gws.base.action.Object):
    """Project information action"""

    @gws.ext.command.api('projectInfo')
    def info(self, req: gws.IWebRequester, p: gws.Request) -> InfoResponse:
        """Return the project configuration"""

        project = req.require_project(p.projectUid)

        locale_uid = p.localeUid
        if locale_uid not in project.localeUids:
            locale_uid = project.localeUids[0]

        return InfoResponse(
            project=gws.props(project, req.user),
            locale=gws.lib.intl.locale(locale_uid),
            user=None if req.user.isGuest else gws.props(req.user, req.user))
