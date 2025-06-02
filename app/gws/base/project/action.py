from typing import Optional

import gws
import gws.base.action
import gws.base.auth.user
import gws.lib.intl


gws.ext.new.action('project')


class Config(gws.base.action.Config):
    """Project info action configuration."""

    pass


class Props(gws.base.action.Props):
    pass


class InfoResponse(gws.Response):
    project: gws.ext.props.project
    locale: gws.Locale
    user: Optional[gws.base.auth.user.Props]


class Object(gws.base.action.Object):
    """Project information action"""

    @gws.ext.command.api('projectInfo')
    def info(self, req: gws.WebRequester, p: gws.Request) -> InfoResponse:
        """Return the project configuration"""

        project = req.user.require_project(p.projectUid)

        return InfoResponse(
            project=gws.props_of(project, req.user),
            locale=gws.lib.intl.locale(p.localeUid, project.localeUids),
            user=None if req.user.isGuest else gws.props_of(req.user, req.user),
        )
